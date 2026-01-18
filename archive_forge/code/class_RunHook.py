from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import collections
import os
import re
import sys
import types
import uuid
import argcomplete
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import backend
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import pkg_resources
import six
class RunHook(object):
    """Encapsulates a function to be run before or after command execution.

  The function should take **kwargs so that more things can be passed to the
  functions in the future.
  """

    def __init__(self, func, include_commands=None, exclude_commands=None):
        """Constructs the hook.

    Args:
      func: function, The function to run.
      include_commands: str, A regex for the command paths to run.  If not
        provided, the hook will be run for all commands.
      exclude_commands: str, A regex for the command paths to exclude.  If not
        provided, nothing will be excluded.
    """
        self.__func = func
        self.__include_commands = include_commands if include_commands else '.*'
        self.__exclude_commands = exclude_commands

    def Run(self, command_path):
        """Runs this hook if the filters match the given command.

    Args:
      command_path: str, The calliope command path for the command that was run.

    Returns:
      bool, True if the hook was run, False if it did not match.
    """
        if not re.match(self.__include_commands, command_path):
            return False
        if self.__exclude_commands and re.match(self.__exclude_commands, command_path):
            return False
        self.__func(command_path=command_path)
        return True