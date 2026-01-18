from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.calliope import walker_util
def DisplayCompletions(command, out=None):
    """Displays the static tab completion data on out.

  The static completion data is a shell script containing variable definitons
  of the form {_COMPLETIONS_PREFIX}{COMMAND.PATH} for each dotted command path.

  Args:
    command: dict, The tree (nested dict) of command/group names.
    out: stream, The output stream, sys.stdout if None.
  """

    def ConvertPathToIdentifier(path):
        return _COMPLETIONS_PREFIX + '__'.join(path).replace('-', '_')

    def WalkCommandTree(command, prefix):
        """Visit each command and group in the CLI command tree.

    Args:
      command: dict, The tree (nested dict) of command/group names.
      prefix: [str], The subcommand arg prefix.
    """
        name = command.get(_LOOKUP_INTERNAL_NAME)
        args = prefix + [name]
        commands = command.get(cli_tree.LOOKUP_COMMANDS, [])
        groups = command.get(cli_tree.LOOKUP_GROUPS, [])
        names = []
        for c in commands + groups:
            names.append(c.get(_LOOKUP_INTERNAL_NAME, c))
        if names:
            flags = command.get(_LOOKUP_INTERNAL_FLAGS, [])
            if prefix:
                out.write('{identifier}=({args})\n'.format(identifier=ConvertPathToIdentifier(args), args=' '.join(names + flags)))
            else:
                out.write('{identifier}=({args})\n'.format(identifier=ConvertPathToIdentifier(['-GCLOUD-WIDE-FLAGS-']), args=' '.join(flags)))
                out.write('{identifier}=({args})\n'.format(identifier=ConvertPathToIdentifier(args), args=' '.join(names)))
            for c in commands:
                name = c.get(_LOOKUP_INTERNAL_NAME, c)
                flags = c.get(_LOOKUP_INTERNAL_FLAGS, [])
                out.write('{identifier}=({args})\n'.format(identifier=ConvertPathToIdentifier(args + [name]), args=' '.join(flags)))
        for g in groups:
            WalkCommandTree(g, args)
    if not out:
        out = sys.stdout
    WalkCommandTree(command, [])