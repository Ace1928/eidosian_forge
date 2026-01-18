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
def BackupDirectory(self):
    """Gets the backup directory of this installation if it exists.

    Returns:
      str, The path to the backup directory or None if it does not exist.
    """
    if self.HasBackup():
        return self.__backup_directory
    return None