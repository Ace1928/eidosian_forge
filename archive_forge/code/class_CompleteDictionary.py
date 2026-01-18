import logging
import stevedore
from cliff import command
class CompleteDictionary:
    """dictionary for bash completion
    """

    def __init__(self):
        self._dictionary = {}

    def add_command(self, command, actions):
        optstr = ' '.join((opt for action in actions for opt in action.option_strings))
        dicto = self._dictionary
        last_cmd = command[-1]
        for subcmd in command[:-1]:
            subdata = dicto.get(subcmd)
            if isinstance(subdata, str):
                subdata += ' ' + last_cmd
                dicto[subcmd] = subdata
                last_cmd = subcmd + '_' + last_cmd
            else:
                dicto = dicto.setdefault(subcmd, {})
        dicto[last_cmd] = optstr

    def get_commands(self):
        return ' '.join((k for k in sorted(self._dictionary.keys())))

    def _get_data_recurse(self, dictionary, path):
        ray = []
        keys = sorted(dictionary.keys())
        for cmd in keys:
            name = path + '_' + cmd if path else cmd
            value = dictionary[cmd]
            if isinstance(value, str):
                ray.append((name, value))
            else:
                cmdlist = ' '.join(sorted(value.keys()))
                ray.append((name, cmdlist))
                ray += self._get_data_recurse(value, name)
        return ray

    def get_data(self):
        return sorted(self._get_data_recurse(self._dictionary, ''))