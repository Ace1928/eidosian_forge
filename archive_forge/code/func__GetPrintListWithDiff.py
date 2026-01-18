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
def _GetPrintListWithDiff(self):
    """Helper method that computes a diff and returns a list of diffs to print.

    Returns:
      List of snapshots.ComponentDiffs for the List method as well as the
      current and latest version strings.
    """
    try:
        _, diff = self._GetStateAndDiff(command_path='components.list')
    except snapshots.IncompatibleSchemaVersionError as e:
        self._ReinstallOnError(e)
        return ([], None, None)
    latest_msg = 'The latest available version is: ' if not self.__fixed_version else None
    current_version, latest_version = self._PrintVersions(diff, latest_msg=latest_msg)
    to_print = diff.AvailableUpdates() + diff.Removed() + diff.AvailableToInstall() + diff.UpToDate()
    return (to_print, current_version, latest_version)