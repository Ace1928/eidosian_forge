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
def ComponentSnapshot(self):
    """Loads the local ComponentSnapshot for this component.

    Returns:
      The snapshots.ComponentSnapshot for this component.
    """
    return snapshots.ComponentSnapshot.FromFile(self.snapshot_file)