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
def CreateStagingFromDownload(self, url, progress_callback=None):
    """Creates a new staging area from a fresh download of the Cloud SDK.

    Args:
      url: str, The url to download the new SDK from.
      progress_callback: f(float), A function to call with the fraction of
        completeness.

    Returns:
      An InstallationState object for the new install.

    Raises:
      installers.URLFetchError: If the new SDK could not be downloaded.
      InvalidDownloadError: If the new SDK was malformed.
    """
    self._ClearStaging()
    with file_utils.TemporaryDirectory() as t:
        download_dir = os.path.join(t, '.download')
        extract_dir = os.path.join(t, '.extract')
        installers.DownloadAndExtractTar(url, download_dir, extract_dir, progress_callback=progress_callback, command_path='components.reinstall')
        files = os.listdir(extract_dir)
        if len(files) != 1:
            raise InvalidDownloadError()
        sdk_root = os.path.join(extract_dir, files[0])
        file_utils.MoveDir(sdk_root, self.__sdk_staging_root)
    staging_sdk = InstallationState(self.__sdk_staging_root)
    staging_sdk._CreateStateDir()
    self.CopyMachinePropertiesTo(staging_sdk)
    return staging_sdk