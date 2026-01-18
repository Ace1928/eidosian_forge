from __future__ import annotations
import collections
import types
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Deque
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
import weakref
from .. import exc
from .. import util
def _collection_gced(ref: weakref.ref[Any]) -> None:
    if not _collection_to_key or ref not in _collection_to_key:
        return
    ref = cast('weakref.ref[RefCollection[EventTarget]]', ref)
    listener_to_key = _collection_to_key.pop(ref)
    for key in listener_to_key.values():
        if key in _key_to_collection:
            dispatch_reg = _key_to_collection[key]
            dispatch_reg.pop(ref)
            if not dispatch_reg:
                _key_to_collection.pop(key)