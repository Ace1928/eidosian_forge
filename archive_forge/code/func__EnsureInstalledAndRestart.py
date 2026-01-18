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
def _EnsureInstalledAndRestart(self, components, msg=None, command=None):
    """Installs the given components if necessary and then restarts gcloud.

    Args:
      components: [str], The components that must be installed.
      msg: str, A custom message to print.
      command: str, the command to run, if not `gcloud`

    Returns:
      bool, True if the components were already installed.  If installation must
      occur, this method never returns because gcloud is reinvoked after the
      update is done.

    Raises:
      MissingRequiredComponentsError: If the components are not installed and
      the user chooses not to install them.
    """
    current_state = self._GetInstallState()
    missing_components = set(components) - set(current_state.InstalledComponents())
    if not missing_components:
        return True
    missing_components_list_str = ', '.join(missing_components)
    if not msg:
        msg = 'This action requires the installation of components: [{components}]'.format(components=missing_components_list_str)
    self.__Write(log.status, msg, word_wrap=True)
    try:
        restart_args = ['components', 'install'] + list(missing_components)
        if not self.Install(components, throw_if_unattended=True, restart_args=restart_args):
            raise MissingRequiredComponentsError("The following components are required to run this command, but are not\ncurrently installed:\n  [{components_list}]\n\nTo install them, re-run the command and choose 'yes' at the installation\nprompt, or run:\n  $ gcloud components install {components}\n\n".format(components_list=missing_components_list_str, components=' '.join(missing_components)))
    except SystemExit:
        self.__Write(log.status, 'Installing component in a new window.\n\nPlease re-run this command when installation is complete.\n    $ {0}'.format(' '.join(['gcloud'] + argv_utils.GetDecodedArgv()[1:])))
        raise
    RestartCommand(command)