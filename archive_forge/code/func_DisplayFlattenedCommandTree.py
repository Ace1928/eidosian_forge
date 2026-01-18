from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.calliope import walker_util
def DisplayFlattenedCommandTree(command, out=None):
    """Displays the commands in the command tree in sorted order on out.

  Args:
    command: dict, The tree (nested dict) of command/group names.
    out: stream, The output stream, sys.stdout if None.
  """

    def WalkCommandTree(commands, command, args):
        """Visit each command and group in the CLI command tree.

    Each command line is added to the commands list.

    Args:
      commands: [str], The list of command strings.
      command: dict, The tree (nested dict) of command/group names.
      args: [str], The subcommand arg prefix.
    """
        args_next = args + [command[_LOOKUP_INTERNAL_NAME]]
        if commands:
            commands.append(' '.join(args_next))
        else:
            commands.append(' '.join(args_next + command.get(_LOOKUP_INTERNAL_FLAGS, [])))
        if cli_tree.LOOKUP_COMMANDS in command:
            for c in command[cli_tree.LOOKUP_COMMANDS]:
                name = c.get(_LOOKUP_INTERNAL_NAME, c)
                flags = c.get(_LOOKUP_INTERNAL_FLAGS, [])
                commands.append(' '.join(args_next + [name] + flags))
        if cli_tree.LOOKUP_GROUPS in command:
            for g in command[cli_tree.LOOKUP_GROUPS]:
                WalkCommandTree(commands, g, args_next)
    commands = []
    WalkCommandTree(commands, command, [])
    if not out:
        out = sys.stdout
    out.write('\n'.join(sorted(commands)) + '\n')