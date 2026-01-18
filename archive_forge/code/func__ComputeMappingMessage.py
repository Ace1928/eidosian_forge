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
def _ComputeMappingMessage(self, command, commands_map, components_map, components=None):
    """Returns error message containing correct command mapping.

    Checks the user-provided command to see if it maps to one we support for
    their package manager. If it does, compute error message to let the user
    know why their command did not work and provide them with an alternate,
    accurate command to run. If we do not support the given command/component
    combination for their package manager, provide user with instructions to
    change their package manager.

    Args:
      command: str, Command from user input, to be mapped against
        commands_mapping.yaml
      commands_map: dict, Contains mappings from commands_mapping.yaml
      components_map: dict, Contains mappings from components_mapping.yaml
      components: str list, Component from user input, to be mapped against
        component_commands.yaml

    Returns:
      str, The compiled error message.
    """
    final_message = ''
    unavailable = 'unavailable'
    update_all = 'update-all'
    unavailable_components = None
    mapped_components = None
    not_components = None
    mapped_packages = None
    correct_command = commands_map[command]
    if components:
        not_components = [component for component in components if component not in components_map]
        unavailable_components = [component for component in components if components_map.get(component) == unavailable]
    if command == update_all:
        mapped_packages = [component for component in set(components_map.values()) if component != unavailable]
    else:
        mapped_components = [component for component in components if component not in unavailable_components and component not in not_components]
        mapped_packages = [components_map[component] for component in mapped_components]
    if mapped_packages:
        correct_command = correct_command.format(package=' '.join(mapped_packages))
    if mapped_packages or not components:
        final_message += '\nYou cannot perform this action because the Google Cloud CLI component manager \nis disabled for this installation. You can run the following command \nto achieve the same result for this installation: \n\n{correct_command}\n\n'.format(correct_command=correct_command)
    if unavailable_components:
        final_message += '\nThe {component} component(s) is unavailable through the packaging system \nyou are currently using. Please consider using a separate installation \nof the Google Cloud CLI created through the default mechanism described at: \n\n{doc_url} \n\n'.format(component=', '.join(unavailable_components), doc_url=config.INSTALLATION_CONFIG.documentation_url)
    if not_components:
        final_message += '"{component}" are not valid component name(s).\n'.format(component=', '.join(not_components))
    return final_message