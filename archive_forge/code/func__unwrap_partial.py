from abc import get_cache_token
from collections import namedtuple
from reprlib import recursive_repr
from _thread import RLock
from types import GenericAlias
def _unwrap_partial(func):
    while isinstance(func, partial):
        func = func.func
    return func