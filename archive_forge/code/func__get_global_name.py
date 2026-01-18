import copy
import abc
import logging
import six
def _get_global_name(self, path):
    if path:
        state = path.pop(0)
        with self.machine(state):
            return self._get_global_name(path)
    else:
        return self.machine.get_global_name()