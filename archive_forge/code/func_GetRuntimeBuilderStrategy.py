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
def GetRuntimeBuilderStrategy(release_track):
    """Gets the appropriate strategy to use for runtime builders.

  Depends on the release track (beta or GA; alpha is not supported) and whether
  the hidden `app/use_runtime_builders` configuration property is set (in which
  case it overrides).

  Args:
    release_track: the base.ReleaseTrack that determines the default strategy.

  Returns:
    The RuntimeBuilderStrategy to use.

  Raises:
    ValueError: if the release track is not supported (and there is no property
      override set).
  """
    if properties.VALUES.app.use_runtime_builders.Get() is not None:
        if properties.VALUES.app.use_runtime_builders.GetBool():
            return runtime_builders.RuntimeBuilderStrategy.ALWAYS
        else:
            return runtime_builders.RuntimeBuilderStrategy.NEVER
    if release_track is base.ReleaseTrack.GA:
        return runtime_builders.RuntimeBuilderStrategy.ALLOWLIST_GA
    elif release_track is base.ReleaseTrack.BETA:
        return runtime_builders.RuntimeBuilderStrategy.ALLOWLIST_BETA
    else:
        raise ValueError('Unrecognized release track [{}]'.format(release_track))