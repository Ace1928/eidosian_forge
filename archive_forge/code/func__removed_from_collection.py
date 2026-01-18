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
def _removed_from_collection(event_key: _EventKey[_ET], owner: RefCollection[_ET]) -> None:
    key = event_key._key
    dispatch_reg = _key_to_collection[key]
    listen_ref = weakref.ref(event_key._listen_fn)
    owner_ref = owner.ref
    dispatch_reg.pop(owner_ref, None)
    if not dispatch_reg:
        del _key_to_collection[key]
    if owner_ref in _collection_to_key:
        listener_to_key = _collection_to_key[owner_ref]
        listener_to_key.pop(listen_ref)