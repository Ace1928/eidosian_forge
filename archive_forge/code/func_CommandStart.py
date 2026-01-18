from __future__ import absolute_import
from __future__ import unicode_literals
import gcloud
import sys
import json
import os
import platform
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from six.moves import input
def CommandStart(command_name, component_id=None, version=None):
    """Logs that the given command is being executed.

  Args:
    command_name: str, The name of the command being executed.
    component_id: str, The component id that this command belongs to.  Used for
      version information if version was not specified.
    version: str, Directly use this version instead of deriving it from
      component.
  """
    if version is None and component_id:
        version = local_state.InstallationState.VersionForInstalledComponent(component_id)
    metrics.Executions(command_name, version)