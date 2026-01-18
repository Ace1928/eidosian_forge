from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import hashlib
import itertools
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap
import certifi
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import release_notes
from googlecloudsdk.core.updater import snapshots
from googlecloudsdk.core.updater import update_check
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
import six
from six.moves import map  # pylint: disable=redefined-builtin
def _PrintVersions(self, diff, latest_msg=None):
    """Prints the current and latest version.

    Args:
      diff: snapshots.ComponentSnapshotDiff, The snapshot diff we are working
        with.
      latest_msg: str, The message to print when displaying the latest version.
        If None, nothing about the latest version is printed.

    Returns:
      (str, str), The current and latest version strings.
    """
    current_version = config.INSTALLATION_CONFIG.version
    latest_version = diff.latest.version
    self.__Write(log.status, '\nYour current Google Cloud CLI version is: ' + current_version)
    if latest_version and latest_msg:
        self.__Write(log.status, latest_msg + latest_version)
    self.__Write(log.status)
    return (current_version, latest_version)