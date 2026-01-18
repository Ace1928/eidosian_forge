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
def _DoFreshInstall(self, message, no_update, download_url):
    """Do a reinstall of what we have based on a fresh download of the SDK.

    Args:
      message: str, A message to show to the user before the re-installation.
      no_update: bool, True to show the message and tell the user they must
        re-download manually.
      download_url: The URL the Cloud SDK can be downloaded from.

    Returns:
      bool, True if the update succeeded, False if it was cancelled.
    """
    self._CheckIfDisabledAndThrowError()
    if encoding.GetEncodedValue(os.environ, 'CLOUDSDK_REINSTALL_COMPONENTS'):
        self._RaiseReinstallationFailedError()
    if message:
        self.__Write(log.status, msg=message, word_wrap=True)
    if no_update:
        return False
    config.EnsureSDKWriteAccess(self.__sdk_root)
    self._RestartIfUsingBundledPython()
    answer = console_io.PromptContinue(message='\nThe component manager must perform a self update before you can continue.  It and all components will be updated to their latest versions.')
    if not answer:
        return False
    self._ShouldDoFastUpdate(allow_no_backup=False, fast_mode_impossible=True, has_components_to_remove=False)
    install_state = self._GetInstallState()
    try:
        with console_io.ProgressBar(label='Downloading and extracting updated components', stream=log.status) as pb:
            staging_state = install_state.CreateStagingFromDownload(download_url, progress_callback=pb.SetProgress)
    except local_state.Error:
        log.error('An updated Google Cloud CLI failed to download')
        log.debug('Handling re-installation error', exc_info=True)
        self._RaiseReinstallationFailedError()
    installed_component_ids = sorted(install_state.InstalledComponents().keys())
    env = encoding.EncodeEnv(dict(os.environ))
    encoding.SetEncodedValue(env, 'CLOUDSDK_REINSTALL_COMPONENTS', ','.join(installed_component_ids))
    installer_path = os.path.join(staging_state.sdk_root, 'bin', 'bootstrapping', 'install.py')
    p = subprocess.Popen([sys.executable, '-S', installer_path], env=env)
    ret_val = p.wait()
    if ret_val:
        self._RaiseReinstallationFailedError()
    with console_io.ProgressBar(label='Creating backup and activating new installation', stream=log.status) as pb:
        install_state.ReplaceWith(staging_state, pb.SetProgress)
    self.__Write(log.status, '\nComponents updated!\n')
    return True