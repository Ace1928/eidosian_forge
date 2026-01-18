from abc import get_cache_token
from collections import namedtuple
from reprlib import recursive_repr
from _thread import RLock
from types import GenericAlias
def _is_union_type(cls):
    from typing import get_origin, Union
    return get_origin(cls) in {Union, types.UnionType}