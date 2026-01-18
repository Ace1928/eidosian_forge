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
def ClearBackup(self, progress_callback=None):
    """Deletes the current backup if it exists.

    Args:
      progress_callback: f(float), A function to call with the fraction of
        completeness.
    """
    if os.path.isdir(self.__backup_directory):
        file_utils.RmTree(self.__backup_directory)
    if progress_callback:
        progress_callback(1)