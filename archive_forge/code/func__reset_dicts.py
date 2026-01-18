from contextlib import contextmanager
from copy import copy
def _reset_dicts(self, value=None):
    builtins = {'True': True, 'False': False, 'None': None}
    self.dicts = [builtins]
    if value is not None:
        self.dicts.append(value)