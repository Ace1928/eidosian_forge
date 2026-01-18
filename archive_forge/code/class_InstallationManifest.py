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
class InstallationManifest(object):
    """Class to encapsulate the data stored in installation manifest files."""
    MANIFEST_SUFFIX = '.manifest'

    def __init__(self, state_dir, component_id):
        """Creates a new InstallationManifest.

    Args:
      state_dir: str, The directory path where install state is stored.
      component_id: str, The component id that you want to get the manifest for.
    """
        self.state_dir = state_dir
        self.id = component_id
        self.snapshot_file = os.path.join(self.state_dir, component_id + InstallationState.COMPONENT_SNAPSHOT_FILE_SUFFIX)
        self.manifest_file = os.path.join(self.state_dir, component_id + InstallationManifest.MANIFEST_SUFFIX)

    def MarkInstalled(self, snapshot, files):
        """Marks this component as installed with the given snapshot and files.

    This saves the ComponentSnapshot and writes the installed files to a
    manifest so they can be removed later.

    Args:
      snapshot: snapshots.ComponentSnapshot, The snapshot that was the source
        of the install.
      files: list of str, The files that were created by the installation.
    """
        with file_utils.FileWriter(self.manifest_file) as fp:
            for f in _NormalizeFileList(files):
                fp.write(f + '\n')
        snapshot.WriteToFile(self.snapshot_file, component_id=self.id)

    def MarkUninstalled(self):
        """Marks this component as no longer being installed.

    This does not actually uninstall the component, but rather just removes the
    snapshot and manifest.
    """
        for f in [self.manifest_file, self.snapshot_file]:
            if os.path.isfile(f):
                os.remove(f)

    def ComponentSnapshot(self):
        """Loads the local ComponentSnapshot for this component.

    Returns:
      The snapshots.ComponentSnapshot for this component.
    """
        return snapshots.ComponentSnapshot.FromFile(self.snapshot_file)

    def ComponentDefinition(self):
        """Loads the ComponentSnapshot and get the schemas.Component this component.

    Returns:
      The schemas.Component for this component.
    """
        return self.ComponentSnapshot().ComponentFromId(self.id)

    def VersionString(self):
        """Gets the version string of this component as it was installed.

    Returns:
      str, The installed version of this component.
    """
        return self.ComponentDefinition().version.version_string

    def InstalledPaths(self):
        """Gets the list of files and dirs created by installing this component.

    Returns:
      list of str, The files and directories installed by this component.
    """
        with file_utils.FileReader(self.manifest_file) as f:
            files = [line.rstrip() for line in f]
        return files

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