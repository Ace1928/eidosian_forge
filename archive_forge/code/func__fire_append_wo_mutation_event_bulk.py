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
def _fire_append_wo_mutation_event_bulk(self, items, initiator=None, key=NO_KEY):
    if not items:
        return
    if initiator is not False:
        if self.invalidated:
            self._warn_invalidated()
        if self.empty:
            self._reset_empty()
        for item in items:
            self.attr.fire_append_wo_mutation_event(self.owner_state, self.owner_state.dict, item, initiator, key)