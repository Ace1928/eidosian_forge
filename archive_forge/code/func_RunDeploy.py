from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import os
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib import scheduler
from googlecloudsdk.api_lib import tasks
from googlecloudsdk.api_lib.app import build as app_cloud_build
from googlecloudsdk.api_lib.app import deploy_app_command_util
from googlecloudsdk.api_lib.app import deploy_command_util
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import metric_names
from googlecloudsdk.api_lib.app import runtime_builders
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.api_lib.app import version_util
from googlecloudsdk.api_lib.app import yaml_parsing
from googlecloudsdk.api_lib.datastore import index_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.tasks import app_deploy_migration_util
from googlecloudsdk.api_lib.util import exceptions as core_api_exceptions
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import create_util
from googlecloudsdk.command_lib.app import deployables
from googlecloudsdk.command_lib.app import exceptions
from googlecloudsdk.command_lib.app import flags
from googlecloudsdk.command_lib.app import output_helpers
from googlecloudsdk.command_lib.app import source_files_util
from googlecloudsdk.command_lib.app import staging
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def RunDeploy(args, api_client, use_beta_stager=False, runtime_builder_strategy=runtime_builders.RuntimeBuilderStrategy.NEVER, parallel_build=True, flex_image_build_option=FlexImageBuildOptions.ON_CLIENT):
    """Perform a deployment based on the given args.

  Args:
    args: argparse.Namespace, An object that contains the values for the
      arguments specified in the ArgsDeploy() function.
    api_client: api_lib.app.appengine_api_client.AppengineClient, App Engine
      Admin API client.
    use_beta_stager: Use the stager registry defined for the beta track rather
      than the default stager registry.
    runtime_builder_strategy: runtime_builders.RuntimeBuilderStrategy, when to
      use the new CloudBuild-based runtime builders (alternative is old
      externalized runtimes).
    parallel_build: bool, whether to use parallel build and deployment path.
      Only supported in v1beta and v1alpha App Engine Admin API.
    flex_image_build_option: FlexImageBuildOptions, whether a flex deployment
      should upload files so that the server can build the image or build the
      image on client or build the image on client using the buildpacks.

  Returns:
    A dict on the form `{'versions': new_versions, 'configs': updated_configs}`
    where new_versions is a list of version_util.Version, and updated_configs
    is a list of config file identifiers, see yaml_parsing.ConfigYamlInfo.
  """
    project = properties.VALUES.core.project.Get(required=True)
    deploy_options = DeployOptions.FromProperties(runtime_builder_strategy=runtime_builder_strategy, parallel_build=parallel_build, flex_image_build_option=flex_image_build_option)
    with files.TemporaryDirectory() as staging_area:
        stager = _MakeStager(args.skip_staging, use_beta_stager, args.staging_command, staging_area)
        services, configs = deployables.GetDeployables(args.deployables, stager, deployables.GetPathMatchers(), args.appyaml)
        wait_for_stop_version = _CheckIfConfigsContainDispatch(configs)
        service_infos = [d.service_info for d in services]
        flags.ValidateImageUrl(args.image_url, service_infos)
        log.debug('API endpoint: [{endpoint}], API version: [{version}]'.format(endpoint=api_client.client.url, version=api_client.client._VERSION))
        app = _PossiblyCreateApp(api_client, project)
        _RaiseIfStopped(api_client, app)
        if not args.bucket:
            app = _PossiblyRepairApp(api_client, app)
        version_id = args.version or util.GenerateVersionId()
        deployed_urls = output_helpers.DisplayProposedDeployment(app, project, services, configs, version_id, deploy_options.promote, args.service_account, api_client.client._VERSION)
        console_io.PromptContinue(cancel_on_no=True)
        if service_infos:
            metrics.CustomTimedEvent(metric_names.GET_CODE_BUCKET_START)
            code_bucket_ref = args.bucket or flags.GetCodeBucket(app, project)
            metrics.CustomTimedEvent(metric_names.GET_CODE_BUCKET)
            log.debug('Using bucket [{b}].'.format(b=code_bucket_ref.ToUrl()))
            if any([s.RequiresImage() for s in service_infos]):
                deploy_command_util.PossiblyEnableFlex(project)
            all_services = dict([(s.id, s) for s in api_client.ListServices()])
        else:
            code_bucket_ref = None
            all_services = {}
        new_versions = []
        deployer = ServiceDeployer(api_client, deploy_options)
        service_deployed = False
        for service in services:
            if not service_deployed:
                metrics.CustomTimedEvent(metric_names.FIRST_SERVICE_DEPLOY_START)
            new_version = version_util.Version(project, service.service_id, version_id)
            deployer.Deploy(service, new_version, code_bucket_ref, args.image_url, all_services, app.gcrDomain, disable_build_cache=not args.cache, wait_for_stop_version=wait_for_stop_version, flex_image_build_option=flex_image_build_option, ignore_file=args.ignore_file, service_account=args.service_account)
            new_versions.append(new_version)
            log.status.Print('Deployed service [{0}] to [{1}]'.format(service.service_id, deployed_urls[service.service_id]))
            if not service_deployed:
                metrics.CustomTimedEvent(metric_names.FIRST_SERVICE_DEPLOY)
            service_deployed = True
    if configs:
        metrics.CustomTimedEvent(metric_names.UPDATE_CONFIG_START)
        for config in configs:
            message = 'Updating config [{config}]'.format(config=config.name)
            with progress_tracker.ProgressTracker(message):
                if config.name == 'dispatch':
                    api_client.UpdateDispatchRules(config.GetRules())
                elif config.name == yaml_parsing.ConfigYamlInfo.INDEX:
                    index_api.CreateMissingIndexes(project, config.parsed)
                elif config.name == yaml_parsing.ConfigYamlInfo.QUEUE:
                    RunDeployCloudTasks(config)
                elif config.name == yaml_parsing.ConfigYamlInfo.CRON:
                    RunDeployCloudScheduler(config)
                else:
                    raise ValueError('Unkonwn config [{config}]'.format(config=config.name))
        metrics.CustomTimedEvent(metric_names.UPDATE_CONFIG)
    updated_configs = [c.name for c in configs]
    PrintPostDeployHints(new_versions, updated_configs)
    return {'versions': new_versions, 'configs': updated_configs}