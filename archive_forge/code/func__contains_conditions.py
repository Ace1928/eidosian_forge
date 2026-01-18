import collections
import collections.abc
import operator
import warnings
def _contains_conditions(self):
    for b in self._bindings:
        if b.get('condition') is not None:
            return True
    return False