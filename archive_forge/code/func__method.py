from abc import get_cache_token
from collections import namedtuple
from reprlib import recursive_repr
from _thread import RLock
from types import GenericAlias
def _method(*args, **kwargs):
    method = self.dispatcher.dispatch(args[0].__class__)
    return method.__get__(obj, cls)(*args, **kwargs)