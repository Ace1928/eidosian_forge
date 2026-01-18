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
def _MakeStager(skip_staging, use_beta_stager, staging_command, staging_area):
    """Creates the appropriate stager for the given arguments/release track.

  The stager is responsible for invoking the right local staging depending on
  env and runtime.

  Args:
    skip_staging: bool, if True use a no-op Stager. Takes precedence over other
      arguments.
    use_beta_stager: bool, if True, use a stager that includes beta staging
      commands.
    staging_command: str, path to an executable on disk. If given, use this
      command explicitly for staging. Takes precedence over later arguments.
    staging_area: str, the path to the staging area

  Returns:
    staging.Stager, the appropriate stager for the command
  """
    if skip_staging:
        return staging.GetNoopStager(staging_area)
    elif staging_command:
        command = staging.ExecutableCommand.FromInput(staging_command)
        return staging.GetOverrideStager(command, staging_area)
    elif use_beta_stager:
        return staging.GetBetaStager(staging_area)
    else:
        return staging.GetStager(staging_area)