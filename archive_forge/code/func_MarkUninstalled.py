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
def MarkUninstalled(self):
    """Marks this component as no longer being installed.

    This does not actually uninstall the component, but rather just removes the
    snapshot and manifest.
    """
    for f in [self.manifest_file, self.snapshot_file]:
        if os.path.isfile(f):
            os.remove(f)