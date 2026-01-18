from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import walker
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_projector
class _Command(object):
    """Command/group info.

  Attributes:
    commands: {str:_Command}, The subcommands in a command group.
    flags: [str], Command flag list. Global flags, available to all commands,
      are in the root command flags list.
  """

    def __init__(self, command, parent):
        self.commands = {}
        self.flags = {}
        self._parent = parent
        if parent:
            name = command.name.replace('_', '-')
            parent.commands[name] = self
        args = command.ai
        for arg in args.flag_args:
            for name in arg.option_strings:
                if arg.is_hidden:
                    continue
                if not name.startswith('--'):
                    continue
                if self.__Ancestor(name):
                    continue
                self.__AddFlag(arg, name)
        for arg in args.ancestor_flag_args:
            for name in arg.option_strings:
                if arg.is_global or arg.is_hidden:
                    continue
                if not name.startswith('--'):
                    continue
                self.__AddFlag(arg, name)

    def __AddFlag(self, flag, name):
        choices = 'bool'
        if flag.choices:
            choices = sorted(flag.choices)
            if choices == ['false', 'true']:
                choices = 'bool'
        elif flag.nargs != 0:
            choices = 'dynamic' if getattr(flag, 'completer', None) else 'value'
        self.flags[name] = choices

    def __Ancestor(self, flag):
        """Determines if flag is provided by an ancestor command.

    Args:
      flag: str, The flag name (no leading '-').

    Returns:
      bool, True if flag provided by an ancestor command, false if not.
    """
        command = self._parent
        while command:
            if flag in command.flags:
                return True
            command = command._parent
        return False