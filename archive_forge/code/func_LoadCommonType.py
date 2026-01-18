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
def LoadCommonType(impl_paths, path, release_track, construction_id, is_command, yaml_command_translator=None):
    """Loads a calliope command or group from a file.

  Args:
    impl_paths: [str], A list of file paths to the command implementation for
      this group or command.
    path: [str], A list of group names that got us down to this command group
      with respect to the CLI itself.  This path should be used for things
      like error reporting when a specific element in the tree needs to be
      referenced.
    release_track: ReleaseTrack, The release track that we should load.
    construction_id: str, A unique identifier for the CLILoader that is
      being constructed.
    is_command: bool, True if we are loading a command, False to load a group.
    yaml_command_translator: YamlCommandTranslator, An instance of a translator
      to use to load the yaml data.

  Raises:
    CommandLoadFailure: If the command is invalid and cannot be loaded.

  Returns:
    The base._Common class for the command or group.
  """
    implementations = _GetAllImplementations(impl_paths, path, construction_id, is_command, yaml_command_translator)
    return _ExtractReleaseTrackImplementation(impl_paths[0], release_track, implementations)()