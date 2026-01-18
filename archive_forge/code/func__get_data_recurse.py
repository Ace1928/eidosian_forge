import logging
import stevedore
from cliff import command
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