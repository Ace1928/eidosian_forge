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
class _BundledCommand(_Command):
    """Represents a cross-platform command.

  Paths are relative to the Cloud SDK Root directory.

  Attributes:
    _nix_path: str, the path to the executable on Linux and OS X
    _windows_path: str, the path to the executable on Windows
    _component: str or None, the name of the Cloud SDK component which contains
      the executable
    _mapper: fn or None, function that maps a staging invocation to a command.
  """

    def __init__(self, nix_path, windows_path, component=None, mapper=None):
        super(_BundledCommand, self).__init__()
        self._nix_path = nix_path
        self._windows_path = windows_path
        self._component = component
        self._mapper = mapper or None

    @property
    def name(self):
        if platforms.OperatingSystem.Current() is platforms.OperatingSystem.WINDOWS:
            return self._windows_path
        else:
            return self._nix_path

    def GetPath(self):
        """Returns the path to the command.

    Returns:
      str, the path to the command

    Raises:
       NoSdkRootError: if no Cloud SDK root could be found (and therefore the
       command is not installed).
    """
        sdk_root = config.Paths().sdk_root
        if not sdk_root:
            raise NoSdkRootError()
        return os.path.join(sdk_root, self.name)

    def GetArgs(self, descriptor, app_dir, staging_dir, explicit_appyaml=None):
        if self._mapper:
            return self._mapper(self.GetPath(), descriptor, app_dir, staging_dir)
        else:
            return super(_BundledCommand, self).GetArgs(descriptor, app_dir, staging_dir)

    def EnsureInstalled(self):
        if self._component is None:
            return
        msg = 'The component [{component}] is required for staging this application.'.format(component=self._component)
        update_manager.UpdateManager.EnsureInstalledAndRestart([self._component], msg=msg)