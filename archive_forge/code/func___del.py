from __future__ import annotations
import operator
import threading
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from .base import NO_KEY
from .. import exc as sa_exc
from .. import util
from ..sql.base import NO_ARG
from ..util.compat import inspect_getfullargspec
from ..util.typing import Protocol
def __del(collection, item, _sa_initiator, key):
    """Run del events.

    This event occurs before the collection is actually mutated, *except*
    in the case of a pop operation, in which case it occurs afterwards.
    For pop operations, the __before_pop hook is called before the
    operation occurs.

    """
    if _sa_initiator is not False:
        executor = collection._sa_adapter
        if executor:
            executor.fire_remove_event(item, _sa_initiator, key=key)