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
@staticmethod
def PerformUpdateCheck(command_path, force=False):
    """Checks to see if a new snapshot has been released periodically.

    This method can be called as often as you'd like.  It will only actually
    check the server for updates if a certain amount of time has elapsed since
    the last check (or if force is True).  If updates are available, to any
    installed components, it will print a notification message.

    Args:
      command_path: str, The '.' separated path of the command that is currently
        being run (i.e. gcloud.foo.bar).
      force: bool, True to force a server check for updates, False to check only
        if the update frequency has expired.
    """
    if config.INSTALLATION_CONFIG.disable_updater or properties.VALUES.component_manager.disable_update_check.GetBool():
        log.debug('SDK update checks are disabled.')
        return
    platform = platforms.Platform.Current()
    manager = UpdateManager(platform_filter=platform, warn=False)
    manager._PerformUpdateCheck(command_path, force=force)