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
def _CheckIfDisabledAndThrowError(self, components=None, command=None):
    """Checks if updater is disabled. If so, raises UpdaterDisabledError.

    The updater is disabled for installations that come from other package
    managers like apt-get or if the current user does not have permission
    to create or delete files in the SDK root directory. If disabled, raises
    UpdaterDisabledError either with the default message, or an error message
    from _ComputeMappingMessage if a command was passed in.

    Args:
      components: str list, Component from user input, to be mapped against
        component_commands.yaml
      command: str, Command from user input, to be mapped against
        command_mapping.yaml

    Raises:
      UpdaterDisabledError: If the updater is disabled.
    """
    default_message = 'You cannot perform this action because this Google Cloud CLI installation is managed by an external package manager.\nPlease consider using a separate installation of the Google Cloud CLI created through the default mechanism described at: {doc_url}\n'.format(doc_url=config.INSTALLATION_CONFIG.documentation_url)
    if config.INSTALLATION_CONFIG.disable_updater:
        if not command:
            raise UpdaterDisabledError(default_message)
        commands_map = self._GetMappingFile(filename='command_mapping.yaml')
        components_map = self._GetMappingFile(filename='component_mapping.yaml')
        if not (components_map and commands_map):
            raise UpdaterDisabledError(default_message)
        mapping_message = self._ComputeMappingMessage(command, commands_map, components_map, components)
        raise UpdaterDisabledError(mapping_message)