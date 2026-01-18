from collections import defaultdict
from operator import itemgetter
from time import time
from typing import TYPE_CHECKING, Any, DefaultDict, Iterable
from weakref import WeakKeyDictionary
class object_ref:
    """Inherit from this class to a keep a record of live instances"""
    __slots__ = ()

    def __new__(cls, *args: Any, **kwargs: Any) -> 'Self':
        obj = object.__new__(cls)
        live_refs[cls][obj] = time()
        return obj