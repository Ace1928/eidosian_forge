from functools import partial
import importlib
import itertools
import numbers
from six import string_types, iteritems
from ..core import Machine
from .nesting import HierarchicalMachine
def _is_auto_transition(self, event):
    if event.name.startswith('to_') and len(event.transitions) == len(self.states):
        state_name = event.name[len('to_'):]
        try:
            _ = self.get_state(state_name)
            return True
        except ValueError:
            pass
    return False