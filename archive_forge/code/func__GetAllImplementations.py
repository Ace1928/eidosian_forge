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
def _GetAllImplementations(impl_paths, path, construction_id, is_command, yaml_command_translator):
    """Gets all the release track command implementations.

  Can load both python and yaml modules.

  Args:
    impl_paths: [str], A list of file paths to the command implementation for
      this group or command.
    path: [str], A list of group names that got us down to this command group
      with respect to the CLI itself.  This path should be used for things
      like error reporting when a specific element in the tree needs to be
      referenced.
    construction_id: str, A unique identifier for the CLILoader that is
      being constructed.
    is_command: bool, True if we are loading a command, False to load a group.
    yaml_command_translator: YamlCommandTranslator, An instance of a translator
      to use to load the yaml data.

  Raises:
    CommandLoadFailure: If the command is invalid and cannot be loaded.

  Returns:
    [(func->base._Common, [base.ReleaseTrack])], A list of tuples that can be
    passed to _ExtractReleaseTrackImplementation. Each item in this list
    represents a command implementation. The first element is a function that
    returns the implementation, and the second element is a list of release
    tracks it is valid for.
  """
    implementations = []
    for impl_file in impl_paths:
        if impl_file.endswith('.yaml'):
            if not is_command:
                raise CommandLoadFailure('.'.join(path), Exception('Command groups cannot be implemented in yaml'))
            if _IsCommandWithPartials(impl_file, path):
                data = _LoadCommandWithPartials(impl_file, path)
            else:
                data = _CustomLoadYamlFile(impl_file)
            implementations.extend(_ImplementationsFromYaml(path, data, yaml_command_translator))
        else:
            module = _GetModuleFromPath(impl_file, path, construction_id)
            implementations.extend(_ImplementationsFromModule(module.__file__, list(module.__dict__.values()), is_command=is_command))
    return implementations