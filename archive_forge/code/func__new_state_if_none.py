from __future__ import annotations
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import base
from . import collections
from . import exc
from . import interfaces
from . import state
from ._typing import _O
from .attributes import _is_collection_attribute_impl
from .. import util
from ..event import EventTarget
from ..util import HasMemoized
from ..util.typing import Literal
from ..util.typing import Protocol
def _new_state_if_none(self, instance: _O) -> Union[Literal[False], InstanceState[_O]]:
    """Install a default InstanceState if none is present.

        A private convenience method used by the __init__ decorator.

        """
    if hasattr(instance, self.STATE_ATTR):
        return False
    elif self.class_ is not instance.__class__ and self.is_mapped:
        return self._subclass_manager(instance.__class__)._new_state_if_none(instance)
    else:
        state = self._state_constructor(instance, self)
        self._state_setter(instance, state)
        return state