import sys
import warnings
from contextlib import contextmanager
from functools import lru_cache as _lru_cache
from typing import Any
from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule
class OptEinsumModule(PropModule):

    def __init__(self, m, name):
        super().__init__(m, name)
    global enabled
    enabled = ContextProp(_get_enabled, _set_enabled)
    global strategy
    strategy = None
    if is_available():
        strategy = ContextProp(_get_strategy, _set_strategy)