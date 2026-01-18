from functools import partial
import importlib
import itertools
import numbers
from six import string_types, iteritems
from ..core import Machine
from .nesting import HierarchicalMachine
def _convert_states(self, root):
    key = 'states' if getattr(self, 'scoped', self) == self else 'children'
    root[key] = []
    for state_name, state in self.states.items():
        s_def = _convert(state, self.state_attributes, self.format_references)
        if isinstance(state_name, Enum):
            s_def['name'] = state_name.name
        else:
            s_def['name'] = state_name
        if getattr(state, 'states', []):
            with self(state_name):
                self._convert_states_and_transitions(s_def)
        root[key].append(s_def)