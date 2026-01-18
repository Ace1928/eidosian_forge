from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import json
import operator
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.app import build as app_cloud_build
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import exceptions
from googlecloudsdk.api_lib.app import instances_util
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.api_lib.app import region_util
from googlecloudsdk.api_lib.app import service_util
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.api_lib.app import version_util
from googlecloudsdk.api_lib.app.api import appengine_api_client_base
from googlecloudsdk.api_lib.cloudbuild import logs as cloudbuild_logs
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.third_party.appengine.admin.tools.conversion import convert_yaml
import six
from six.moves import filter  # pylint: disable=redefined-builtin
from six.moves import map  # pylint: disable=redefined-builtin
def DeployService(self, service_name, version_id, service_config, manifest, build, extra_config_settings=None, service_account_email=None):
    """Updates and deploys new app versions.

    Args:
      service_name: str, The service to deploy.
      version_id: str, The version of the service to deploy.
      service_config: AppInfoExternal, Service info parsed from a service yaml
        file.
      manifest: Dictionary mapping source files to Google Cloud Storage
        locations.
      build: BuildArtifact, a wrapper which contains either the build
        ID for an in-progress parallel build, the name of the container image
        for a serial build, or the options for creating a build elsewhere. Not
        present during standard deploys.
      extra_config_settings: dict, client config settings to pass to the server
        as beta settings.
      service_account_email: Identity of this deployed version. If not set, the
        Admin API will fall back to use the App Engine default appspot service
        account.

    Returns:
      The Admin API Operation, unfinished.

    Raises:
      apitools_exceptions.HttpNotFoundError if build ID doesn't exist
    """
    operation = self._CreateVersion(service_name, version_id, service_config, manifest, build, extra_config_settings, service_account_email)
    message = 'Updating service [{service}]'.format(service=service_name)
    if service_config.env in [env.FLEX, env.MANAGED_VMS]:
        message += ' (this may take several minutes)'
    operation_metadata_type = self._ResolveMetadataType()
    if build and build.IsBuildOptions():
        if not operation_metadata_type:
            log.warning('Unable to determine build from Operation metadata. Skipping log streaming')
        else:
            poller = operations_util.AppEngineOperationBuildPoller(self.client.apps_operations, operation_metadata_type)
            operation = operations_util.WaitForOperation(self.client.apps_operations, operation, message=message, poller=poller)
            build_id = operations_util.GetBuildFromOperation(operation, operation_metadata_type)
            if build_id:
                build = app_cloud_build.BuildArtifact.MakeBuildIdArtifact(build_id)
    if build and build.IsBuildId():
        try:
            build_ref = resources.REGISTRY.Parse(build.identifier, params={'projectId': properties.VALUES.core.project.GetOrFail}, collection='cloudbuild.projects.builds')
            cloudbuild_logs.CloudBuildClient().Stream(build_ref, out=log.status)
        except apitools_exceptions.HttpNotFoundError:
            region = util.ConvertToCloudRegion(self.GetApplication().locationId)
            build_ref = resources.REGISTRY.Create(collection='cloudbuild.projects.locations.builds', projectsId=properties.VALUES.core.project.GetOrFail, locationsId=region, buildsId=build.identifier)
            cloudbuild_logs.CloudBuildClient().Stream(build_ref, out=log.status)
    done_poller = operations_util.AppEngineOperationPoller(self.client.apps_operations, operation_metadata_type)
    return operations_util.WaitForOperation(self.client.apps_operations, operation, message=message, poller=done_poller)