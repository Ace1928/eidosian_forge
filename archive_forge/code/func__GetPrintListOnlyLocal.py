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
def _GetPrintListOnlyLocal(self):
    """Helper method that gets a list of locally installed components to print.

    Returns:
      List of snapshots.ComponentInfos for the List method as well as the
      current version string.
    """
    install_state = self._GetInstallState()
    to_print = install_state.Snapshot().CreateComponentInfos(platform_filter=self.__platform_filter)
    if self._EnableFallback():
        native_ids = set((c.id for c in to_print))
        darwin_x86_64_all = install_state.Snapshot().CreateComponentInfos(platform_filter=self.DARWIN_X86_64)
        to_print_x86_64 = (c for c in darwin_x86_64_all if c.id not in native_ids)
        to_print.extend(to_print_x86_64)
    current_version = config.INSTALLATION_CONFIG.version
    self.__Write(log.status, '\nYour current Google Cloud CLI version is: ' + current_version)
    return (to_print, current_version)