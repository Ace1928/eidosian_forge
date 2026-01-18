from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import importlib
import os
import re
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import command_release_tracks
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import pkg_resources
from ruamel import yaml
import six
def _IsCommandWithPartials(impl_file, path):
    """Checks if the YAML file is a command with partials.

  Args:
    impl_file: file path to the main YAML command implementation.
    path: [str], A list of group names that got us down to this command group
      with respect to the CLI itself.  This path should be used for things
      like error reporting when a specific element in the tree needs to be
      referenced.

  Raises:
    CommandLoadFailure: If the command is invalid and should not be loaded.

  Returns:
    Whether or not it is a valid command with partials to load.
  """
    found_partial_token = False
    with pkg_resources.GetFileTextReaderByLine(impl_file) as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line == f'{PARTIALS_ATTRIBUTE}: true':
                found_partial_token = True
            elif found_partial_token:
                raise CommandLoadFailure('.'.join(path), Exception(f'Command with {PARTIALS_ATTRIBUTE} attribute cannot have extra content'))
            else:
                break
    return found_partial_token