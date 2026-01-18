from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import os
import re
import threading
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import properties_file
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
class _FlagOverrideStack(object):
    """Class representing a stack of configuration flag values or `None`s.

  Each time a command line is parsed (the original, and any from commands
  calling other commands internally), the new value for the --configuration
  flag is added to the top of the stack (if it is provided).  This is used for
  resolving the currently active configuration.

  We need to do quick parsing of the args here because things like logging are
  used before argparse parses the command line and logging needs properties.
  We scan the command line for the --configuration flag to get the active
  configuration before any of that starts.
  """

    def __init__(self):
        self._stack = []

    def Push(self, config_flag):
        """Add a new value to the top of the stack."""
        old_flag = self.ActiveConfig()
        self._stack.append(config_flag)
        if old_flag != config_flag:
            ActivePropertiesFile.Invalidate()

    def PushFromArgs(self, args):
        """Parse the args and add the value that was found to the top of the stack.

    Args:
      args: [str], The command line args for this invocation.
    """
        self.Push(_FlagOverrideStack._FindFlagValue(args))

    def Pop(self):
        """Remove the top value from the stack."""
        return self._stack.pop()

    def ActiveConfig(self):
        """Get the top most value on the stack."""
        for value in reversed(self._stack):
            if value:
                return value
        return None

    @classmethod
    def _FindFlagValue(cls, args):
        """Parse the given args to find the value of the --configuration flag.

    Args:
      args: [str], The arguments from the command line to parse

    Returns:
      str, The value of the --configuration flag or None if not found.
    """
        flag = '--configuration'
        flag_eq = flag + '='
        successor = None
        config_flag = None
        for arg in reversed(args):
            if arg == flag:
                config_flag = successor
                break
            if arg.startswith(flag_eq):
                _, config_flag = arg.split('=', 1)
                break
            successor = arg
        return config_flag