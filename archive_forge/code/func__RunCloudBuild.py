from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import random
import string
import time
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py.exceptions import HttpError
from apitools.base.py.exceptions import HttpNotFoundError
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import logs as cb_logs
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.services import enable_api as services_api
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as http_exc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.cloudbuild import execution
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding as encoding_util
import six
def _RunCloudBuild(args, builder, build_args, build_tags=None, output_filter=None, log_location=None, backoff=lambda elapsed: 1, build_region=None):
    """Run a build with a specific builder on Google Cloud Builder.

  Args:
    args: an argparse namespace. All the arguments that were provided to this
      command invocation.
    builder: A paths to builder Docker image.
    build_args: args to be sent to builder
    build_tags: tags to be attached to the build
    output_filter: A list of strings indicating what lines from the log should
      be output. Only lines that start with one of the strings in output_filter
      will be displayed.
    log_location: GCS path to directory where logs will be stored.
    backoff: A function that takes the current elapsed time and returns
      the next sleep length. Both are in seconds.
    build_region: Region to run Cloud Build in.

  Returns:
    A build object that either streams the output or is displayed as a
    link to the build.

  Raises:
    FailedBuildException: If the build is completed and not 'SUCCESS'.
  """
    client = cloudbuild_util.GetClientInstance()
    messages = cloudbuild_util.GetMessagesModule()
    build_config = messages.Build(steps=[messages.BuildStep(name=builder, args=sorted(build_args))], tags=build_tags, timeout='{0}s'.format(args.timeout))
    if log_location:
        gcs_log_ref = resources.REGISTRY.Parse(args.log_location)
        if hasattr(gcs_log_ref, 'object'):
            build_config.logsBucket = 'gs://{0}/{1}'.format(gcs_log_ref.bucket, gcs_log_ref.object)
        else:
            build_config.logsBucket = 'gs://{0}'.format(gcs_log_ref.bucket)
    if hasattr(args, 'cloudbuild_service_account') and args.cloudbuild_service_account:
        if not build_config.logsBucket:
            raise calliope_exceptions.RequiredArgumentException('--log-location', 'Log Location  is required when service account is provided for cloud build')
        build_config.serviceAccount = 'projects/{0}/serviceAccounts/{1}'.format(properties.VALUES.core.project.Get(), args.cloudbuild_service_account)
    if build_region and build_region in AR_TO_CLOUD_BUILD_REGIONS:
        build_region = AR_TO_CLOUD_BUILD_REGIONS[build_region]
    if build_region and build_region in CLOUD_BUILD_REGIONS:
        build, build_ref = _CreateRegionalCloudBuild(build_config, client, messages, build_region)
    else:
        build, build_ref = _CreateCloudBuild(build_config, client, messages)
    if args.async_:
        return build
    mash_handler = execution.MashHandler(execution.GetCancelBuildHandler(client, messages, build_ref))
    with execution_utils.CtrlCSection(mash_handler):
        build = CloudBuildClientWithFiltering(client, messages).StreamWithFilter(build_ref, backoff, output_filter=output_filter)
    if build.status == messages.Build.StatusValueValuesEnum.TIMEOUT:
        log.status.Print('Your build timed out. Use the [--timeout=DURATION] flag to change the timeout threshold.')
    if build.status != messages.Build.StatusValueValuesEnum.SUCCESS:
        raise FailedBuildException(build)
    return build