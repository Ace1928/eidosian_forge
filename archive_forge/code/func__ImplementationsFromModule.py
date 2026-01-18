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
def _ImplementationsFromModule(mod_file, module_attributes, is_command):
    """Gets all the release track command implementations from the module.

  Args:
    mod_file: str, The __file__ attribute of the module resulting from
      importing the file containing a command.
    module_attributes: The __dict__.values() of the module.
    is_command: bool, True if we are loading a command, False to load a group.

  Raises:
    LayoutException: If there is not exactly one type inheriting CommonBase.

  Returns:
    [(func->base._Common, [base.ReleaseTrack])], A list of tuples that can be
    passed to _ExtractReleaseTrackImplementation. Each item in this list
    represents a command implementation. The first element is a function that
    returns the implementation, and the second element is a list of release
    tracks it is valid for.
  """
    commands = []
    groups = []
    for command_or_group in module_attributes:
        if getattr(command_or_group, 'IS_COMMAND', False):
            commands.append(command_or_group)
        elif getattr(command_or_group, 'IS_COMMAND_GROUP', False):
            groups.append(command_or_group)
    if is_command:
        if groups:
            raise LayoutException('You cannot define groups [{0}] in a command file: [{1}]'.format(', '.join([g.__name__ for g in groups]), mod_file))
        if not commands:
            raise LayoutException('No commands defined in file: [{0}]'.format(mod_file))
        commands_or_groups = commands
    else:
        if commands:
            raise LayoutException('You cannot define commands [{0}] in a command group file: [{1}]'.format(', '.join([c.__name__ for c in commands]), mod_file))
        if not groups:
            raise LayoutException('No command groups defined in file: [{0}]'.format(mod_file))
        commands_or_groups = groups
    return [(lambda c=c: c, c.ValidReleaseTracks()) for c in commands_or_groups]