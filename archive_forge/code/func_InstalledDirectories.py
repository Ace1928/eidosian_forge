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
def InstalledDirectories(self):
    """Gets the set of directories created by installing this component.

    Returns:
      set(str), The directories installed by this component.
    """
    with file_utils.FileReader(self.manifest_file) as f:
        dirs = set()
        for line in f:
            norm_file_path = os.path.dirname(line.rstrip())
            prev_file = norm_file_path + '/'
            while len(prev_file) > len(norm_file_path) and norm_file_path:
                dirs.add(norm_file_path)
                prev_file = norm_file_path
                norm_file_path = os.path.dirname(norm_file_path)
    return dirs