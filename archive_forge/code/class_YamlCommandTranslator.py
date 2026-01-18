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
class YamlCommandTranslator(six.with_metaclass(abc.ABCMeta, object)):
    """An interface to implement when registering a custom command loader."""

    @abc.abstractmethod
    def Translate(self, path, command_data):
        """Translates a yaml command into a calliope command.

    Args:
      path: [str], A list of group names that got us down to this command group
        with respect to the CLI itself.  This path should be used for things
        like error reporting when a specific element in the tree needs to be
        referenced.
      command_data: dict, The parsed contents of the command spec from the
        yaml file that corresponds to the release track being loaded.

    Returns:
      calliope.base.Command, A command class (not instance) that
      implements the spec.
    """
        pass