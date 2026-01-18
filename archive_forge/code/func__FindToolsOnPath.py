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
def _FindToolsOnPath(self, path=None, duplicates=True, other=True):
    """Helper function to find commands matching SDK bin dir on the path."""
    bin_dir = os.path.realpath(os.path.join(self.__sdk_root, UpdateManager.BIN_DIR_NAME))
    if not os.path.exists(bin_dir):
        return set()
    commands = [f for f in os.listdir(bin_dir) if os.path.isfile(os.path.join(bin_dir, f)) and (not f.startswith('.'))]
    duplicates_in_sdk_root = set()
    bad_commands = set()
    for command in commands:
        existing_paths = file_utils.SearchForExecutableOnPath(command, path=path)
        if existing_paths:
            this_tool = os.path.join(bin_dir, command)
            if other:
                bad_commands.update(set((os.path.realpath(f) for f in existing_paths if self.__sdk_root not in os.path.realpath(f))) - set([this_tool]))
            if duplicates:
                duplicates_in_sdk_root.update(set((os.path.realpath(f) for f in existing_paths if self.__sdk_root in os.path.realpath(f))) - set([this_tool]))
    return bad_commands.union(duplicates_in_sdk_root)