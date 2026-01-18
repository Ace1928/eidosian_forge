from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import inspect
from fire import inspectutils
import six
def _GetMaps(name, commands, default_options):
    """Returns sets of subcommands and options for each command.

  Args:
    name: The first token in the commands, also the name of the command.
    commands: A list of all possible commands that tab completion can complete
        to. Each command is a list or tuple of the string tokens that make up
        that command.
    default_options: A dict of options that can be used with any command. Use
        this if there are flags that can always be appended to a command.
  Returns:
    global_options: A set of all options of the first token of the command.
    subcommands_map: A dict storing set of subcommands for each
        command/subcommand.
    options_map: A dict storing set of options for each subcommand.
  """
    global_options = copy.copy(default_options)
    options_map = collections.defaultdict(lambda: copy.copy(default_options))
    subcommands_map = collections.defaultdict(set)
    for command in commands:
        if len(command) == 1:
            if _IsOption(command[0]):
                global_options.add(command[0])
            else:
                subcommands_map[name].add(command[0])
        elif command:
            subcommand = command[-2]
            arg = _FormatForCommand(command[-1])
            if _IsOption(arg):
                args_map = options_map
            else:
                args_map = subcommands_map
            args_map[subcommand].add(arg)
            args_map[subcommand.replace('_', '-')].add(arg)
    return (global_options, options_map, subcommands_map)