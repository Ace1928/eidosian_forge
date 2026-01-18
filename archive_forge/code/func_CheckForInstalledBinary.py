from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import os
from googlecloudsdk.command_lib.util.anthos import structured_messages as sm
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def CheckForInstalledBinary(binary_name, check_hidden=False, custom_message=None, install_if_missing=False):
    """Check if binary is installed and return path or raise error.

  Prefer the installed component over any version found on path.

  Args:
    binary_name: str, name of binary to search for.
    check_hidden: bool, whether to check hidden components for the binary.
    custom_message: str, custom message to used by MissingExecutableException if
      thrown.
    install_if_missing: bool, if true will prompt user to install binary if not
      found.

  Returns:
    Path to executable if found on path or installed component.

  Raises:
    MissingExecutableException: if executable can not be found or can not be
     installed as a component.
  """
    is_component = CheckBinaryComponentInstalled(binary_name, check_hidden)
    if is_component:
        return os.path.join(config.Paths().sdk_bin_path, binary_name)
    path_executable = files.FindExecutableOnPath(binary_name)
    if path_executable:
        return path_executable
    if install_if_missing:
        return InstallBinaryNoOverrides(binary_name, _INSTALL_MISSING_EXEC_PROMPT.format(binary=binary_name))
    raise MissingExecutableException(binary_name, custom_message)