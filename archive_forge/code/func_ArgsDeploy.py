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
def ArgsDeploy(parser):
    """Get arguments for this command.

  Args:
    parser: argparse.ArgumentParser, the parser for this command.
  """
    flags.SERVER_FLAG.AddToParser(parser)
    flags.IGNORE_CERTS_FLAG.AddToParser(parser)
    flags.DOCKER_BUILD_FLAG.AddToParser(parser)
    flags.IGNORE_FILE_FLAG.AddToParser(parser)
    parser.add_argument('--version', '-v', type=flags.VERSION_TYPE, help='The version of the app that will be created or replaced by this deployment.  If you do not specify a version, one will be generated for you.')
    parser.add_argument('--bucket', type=storage_util.BucketReference.FromArgument, help="The Google Cloud Storage bucket used to stage files associated with the deployment. If this argument is not specified, the application's default code bucket is used.")
    parser.add_argument('--service-account', help='The service account that this deployed version will run as. If this argument is not specified, the App Engine default service account will be used for your current deployed version.')
    parser.add_argument('deployables', nargs='*', help='      The yaml files for the services or configurations you want to deploy.\n      If not given, defaults to `app.yaml` in the current directory.\n      If that is not found, attempts to automatically generate necessary\n      configuration files (such as app.yaml) in the current directory.')
    parser.add_argument('--stop-previous-version', action=actions.StoreBooleanProperty(properties.VALUES.app.stop_previous_version), help='      Stop the previously running version when deploying a new version that\n      receives all traffic.\n\n      Note that if the version is running on an instance\n      of an auto-scaled service in the App Engine Standard\n      environment, using `--stop-previous-version` will not work\n      and the previous version will continue to run because auto-scaled service\n      instances are always running.')
    parser.add_argument('--image-url', help='(App Engine flexible environment only.) Deploy with a specific Docker image. Docker url must be from one of the valid Container Registry hostnames.')
    parser.add_argument('--appyaml', help='Deploy with a specific app.yaml that will replace the one defined in the DEPLOYABLE.')
    parser.add_argument('--promote', action=actions.StoreBooleanProperty(properties.VALUES.app.promote_by_default), help='Promote the deployed version to receive all traffic.')
    parser.add_argument('--cache', action='store_true', default=True, help='Enable caching mechanisms involved in the deployment process, particularly in the build step.')
    staging_group = parser.add_mutually_exclusive_group(hidden=True)
    staging_group.add_argument('--skip-staging', action='store_true', default=False, help='THIS ARGUMENT NEEDS HELP TEXT.')
    staging_group.add_argument('--staging-command', help='THIS ARGUMENT NEEDS HELP TEXT.')