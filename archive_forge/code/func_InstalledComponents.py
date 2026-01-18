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
@_RaisesPermissionsError
def InstalledComponents(self):
    """Gets all the components that are currently installed.

    Returns:
      A dictionary of component id string to InstallationManifest.
    """
    snapshot_files = self._FilesForSuffix(InstallationState.COMPONENT_SNAPSHOT_FILE_SUFFIX)
    manifests = {}
    for f in snapshot_files:
        component_id = f[:-len(InstallationState.COMPONENT_SNAPSHOT_FILE_SUFFIX)]
        manifests[component_id] = InstallationManifest(self._state_directory, component_id)
    return manifests