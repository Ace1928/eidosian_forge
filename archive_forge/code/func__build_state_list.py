from collections import OrderedDict
import copy
from functools import partial, reduce
import inspect
import logging
from six import string_types
from ..core import State, Machine, Transition, Event, listify, MachineError, EventData
def _build_state_list(state_tree, separator, prefix=None):
    prefix = prefix or []
    res = []
    for key, value in state_tree.items():
        if value:
            res.append(_build_state_list(value, separator, prefix=prefix + [key]))
        else:
            res.append(separator.join(prefix + [key]))
    return res if len(res) > 1 else res[0]