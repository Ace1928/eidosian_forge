from functools import partial
import importlib
import itertools
import numbers
from six import string_types, iteritems
from ..core import Machine
from .nesting import HierarchicalMachine
def _convert_states_and_transitions(self, root):
    state = getattr(self, 'scoped', self)
    if state.initial:
        root['initial'] = state.initial
    if state == self and state.name:
        root['name'] = self.name[:-2]
    self._convert_transitions(root)
    self._convert_states(root)