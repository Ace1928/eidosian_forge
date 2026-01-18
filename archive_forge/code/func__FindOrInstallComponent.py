from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import json
import os.path
import subprocess
import threading
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.command_lib.code import json_stream
from googlecloudsdk.core import config
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files as file_utils
import six
def _FindOrInstallComponent(component_name):
    """Finds the path to a component or install it.

  Args:
    component_name: Name of the component.

  Returns:
    Path to the component. Returns None if the component can't be found.
  """
    if config.Paths().sdk_root and update_manager.UpdateManager.EnsureInstalledAndRestart([component_name]):
        return os.path.join(config.Paths().sdk_root, 'bin', component_name)
    return None