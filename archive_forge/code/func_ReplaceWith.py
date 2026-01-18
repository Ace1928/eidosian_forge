from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import compileall
import errno
import logging
import os
import posixpath
import re
import shutil
import sys
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import snapshots
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
import six
@_RaisesPermissionsError
def ReplaceWith(self, other_install_state, progress_callback=None):
    """Replaces this installation with the given other installation.

    This moves the current installation to the backup directory of the other
    installation.  Then, it moves the entire second installation to replace
    this one on the file system.  The result is that the other installation
    completely replaces the current one, but the current one is snapshotted and
    stored as a backup under the new one (and can be restored later).

    Args:
      other_install_state: InstallationState, The other state with which to
        replace this one.
      progress_callback: f(float), A function to call with the fraction of
        completeness.
    """
    self._CreateStateDir()
    self.ClearBackup()
    self.ClearTrash()
    other_install_state._CreateStateDir()
    other_install_state.ClearBackup()
    file_utils.MoveDir(self.__sdk_root, other_install_state.__backup_directory)
    if progress_callback:
        progress_callback(0.5)
    file_utils.MoveDir(other_install_state.__sdk_root, self.__sdk_root)
    if progress_callback:
        progress_callback(1.0)