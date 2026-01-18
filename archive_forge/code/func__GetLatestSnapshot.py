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
def _GetLatestSnapshot(self, version=None, command_path='unknown'):
    effective_url = self._GetEffectiveSnapshotURL(version=version)
    try:
        return snapshots.ComponentSnapshot.FromURLs(*effective_url.split(','), command_path=command_path)
    except snapshots.URLFetchError:
        if version:
            log.error('The component listing for Google Cloud CLI version [{0}] could not be found.  Make sure this is a valid archived Google Cloud CLI version.'.format(version))
        elif self.__fixed_version:
            log.error('You have configured your Google Cloud CLI installation to be fixed to version [{0}]. Make sure this is a valid archived Google Cloud CLI version.'.format(self.__fixed_version))
        raise