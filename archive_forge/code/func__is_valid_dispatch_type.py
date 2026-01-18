from abc import get_cache_token
from collections import namedtuple
from reprlib import recursive_repr
from _thread import RLock
from types import GenericAlias
def _is_valid_dispatch_type(cls):
    if isinstance(cls, type):
        return True
    from typing import get_args
    return _is_union_type(cls) and all((isinstance(arg, type) for arg in get_args(cls)))