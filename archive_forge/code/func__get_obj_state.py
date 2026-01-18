import copyreg
import functools
import sys
import traceback
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, DefaultDict, List, Optional
import torch
def _get_obj_state(obj):
    getstate_fn = getattr(obj, '__getstate__', None)
    if getstate_fn:
        state = getstate_fn()
    else:
        slots_to_save = copyreg._slotnames(obj.__class__)
        if slots_to_save:
            state = (obj.__dict__, {name: getattr(obj, name) for name in slots_to_save if hasattr(obj, name)})
        else:
            state = obj.__dict__
    return state