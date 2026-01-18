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
def RunOsUpgradeBuild(args, output_filter, instance_uri, release_track):
    """Run a OS Upgrade on Google Cloud Builder.

  Args:
    args: an argparse namespace. All the arguments that were provided to this
      command invocation.
    output_filter: A list of strings indicating what lines from the log should
      be output. Only lines that start with one of the strings in output_filter
      will be displayed.
    instance_uri: instance to be upgraded.
    release_track: release track of the command used. One of - "alpha", "beta"
      or "ga"

  Returns:
    A build object that either streams the output or is displayed as a
    link to the build.

  Raises:
    FailedBuildException: If the build is completed and not 'SUCCESS'.
  """
    del release_track
    project_id = projects_util.ParseProject(properties.VALUES.core.project.GetOrFail())
    _CheckIamPermissions(project_id, frozenset(OS_UPGRADE_ROLES_FOR_CLOUDBUILD_SERVICE_ACCOUNT), frozenset(OS_UPGRADE_ROLES_FOR_COMPUTE_SERVICE_ACCOUNT))
    two_percent = int(args.timeout * 0.02)
    os_upgrade_timeout = args.timeout - min(two_percent, 300)
    os_upgrade_args = []
    AppendArg(os_upgrade_args, 'instance', instance_uri)
    AppendArg(os_upgrade_args, 'source-os', args.source_os)
    AppendArg(os_upgrade_args, 'target-os', args.target_os)
    AppendArg(os_upgrade_args, 'timeout', os_upgrade_timeout, '-{0}={1}s')
    AppendArg(os_upgrade_args, 'client-id', 'gcloud')
    if not args.create_machine_backup:
        AppendArg(os_upgrade_args, 'create-machine-backup', 'false')
    AppendBoolArg(os_upgrade_args, 'auto-rollback', args.auto_rollback)
    AppendBoolArg(os_upgrade_args, 'use-staging-install-media', args.use_staging_install_media)
    AppendArg(os_upgrade_args, 'client-version', config.CLOUD_SDK_VERSION)
    build_tags = ['gce-os-upgrade']
    builder_region = _GetBuilderRegion(_GetOSUpgradeRegion, args)
    builder = _GetBuilder(_OS_UPGRADE_BUILDER_EXECUTABLE, args.docker_image_tag, builder_region)
    return _RunCloudBuild(args, builder, os_upgrade_args, build_tags, output_filter, args.log_location, build_region=builder_region)