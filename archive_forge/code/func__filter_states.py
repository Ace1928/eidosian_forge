import logging
from functools import partial
from collections import defaultdict
from os.path import splitext
from .diagrams_base import BaseGraph
def _filter_states(states, state_names, state_cls, prefix=None):
    prefix = prefix or []
    result = []
    for state in states:
        pref = prefix + [state['name']]
        if 'children' in state:
            state['children'] = _filter_states(state['children'], state_names, state_cls, prefix=pref)
            result.append(state)
        elif getattr(state_cls, 'separator', '_').join(pref) in state_names:
            result.append(state)
    return result