from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import inspect
from fire import inspectutils
import six
def _FishScript(name, commands, default_options=None):
    """Returns a Fish script registering a completion function for the commands.

  Args:
    name: The first token in the commands, also the name of the command.
    commands: A list of all possible commands that tab completion can complete
        to. Each command is a list or tuple of the string tokens that make up
        that command.
    default_options: A dict of options that can be used with any command. Use
        this if there are flags that can always be appended to a command.
  Returns:
    A string which is the Fish script. Source the fish script to enable tab
    completion in Fish.
  """
    default_options = default_options or set()
    global_options, options_map, subcommands_map = _GetMaps(name, commands, default_options)
    fish_source = 'function __fish_using_command\n    set cmd (commandline -opc)\n    for i in (seq (count $cmd) 1)\n        switch $cmd[$i]\n        case "-*"\n        case "*"\n            if [ $cmd[$i] = $argv[1] ]\n                return 0\n            else\n                return 1\n            end\n        end\n    end\n    return 1\nend\n\nfunction __option_entered_check\n    set cmd (commandline -opc)\n    for i in (seq (count $cmd))\n        switch $cmd[$i]\n        case "-*"\n            if [ $cmd[$i] = $argv[1] ]\n                return 1\n            end\n        end\n    end\n    return 0\nend\n\nfunction __is_prev_global\n    set cmd (commandline -opc)\n    set global_options {global_options}\n    set prev (count $cmd)\n\n    for opt in $global_options\n        if [ "--$opt" = $cmd[$prev] ]\n            echo $prev\n            return 0\n        end\n    end\n    return 1\nend\n\n'
    subcommand_template = "complete -c {name} -n '__fish_using_command {command}' -f -a {subcommand}\n"
    flag_template = "complete -c {name} -n '__fish_using_command {command};{prev_global_check} and __option_entered_check --{option}' -l {option}\n"
    prev_global_check = ' and __is_prev_global;'
    for command in set(subcommands_map.keys()).union(set(options_map.keys())):
        for subcommand in subcommands_map[command]:
            fish_source += subcommand_template.format(name=name, command=command, subcommand=subcommand)
        for option in options_map[command].union(global_options):
            check_needed = command != name
            fish_source += flag_template.format(name=name, command=command, prev_global_check=prev_global_check if check_needed else '', option=option.lstrip('--'))
    return fish_source.format(global_options=' '.join(('"{option}"'.format(option=option) for option in global_options)))