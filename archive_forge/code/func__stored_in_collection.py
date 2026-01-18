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
def _stored_in_collection(event_key: _EventKey[_ET], owner: RefCollection[_ET]) -> bool:
    key = event_key._key
    dispatch_reg = _key_to_collection[key]
    owner_ref = owner.ref
    listen_ref = weakref.ref(event_key._listen_fn)
    if owner_ref in dispatch_reg:
        return False
    dispatch_reg[owner_ref] = listen_ref
    listener_to_key = _collection_to_key[owner_ref]
    listener_to_key[listen_ref] = key
    return True