from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
import os
import re
import shutil
import tempfile
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import runtime_registry
from googlecloudsdk.command_lib.app import jarfile
from googlecloudsdk.command_lib.util import java
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
class ExecutableCommand(_Command):
    """Represents a command that the user supplies.

  Attributes:
    _path: str, full path to the executable.
  """

    def __init__(self, path):
        super(ExecutableCommand, self).__init__()
        self._path = path

    @property
    def name(self):
        os.path.basename(self._path)

    def GetPath(self):
        return self._path

    def EnsureInstalled(self):
        pass

    def GetArgs(self, descriptor, app_dir, staging_dir, explicit_appyaml=None):
        if explicit_appyaml:
            return [self.GetPath(), descriptor, app_dir, staging_dir, explicit_appyaml]
        else:
            return [self.GetPath(), descriptor, app_dir, staging_dir]

    @classmethod
    def FromInput(cls, executable):
        """Returns the command corresponding to the user input.

    Could be either of:
    - command on the $PATH or %PATH%
    - full path to executable (absolute or relative)

    Args:
      executable: str, the user-specified staging exectuable to use

    Returns:
      _Command corresponding to the executable

    Raises:
      StagingCommandNotFoundError: if the executable couldn't be found
    """
        try:
            path = files.FindExecutableOnPath(executable)
        except ValueError:
            path = None
        if path:
            return cls(path)
        if os.path.exists(executable):
            return cls(executable)
        raise StagingCommandNotFoundError('The provided staging command [{}] could not be found.'.format(executable))