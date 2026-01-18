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
class _SerializeManager:
    """Provide serialization of a :class:`.ClassManager`.

    The :class:`.InstanceState` uses ``__init__()`` on serialize
    and ``__call__()`` on deserialize.

    """

    def __init__(self, state: state.InstanceState[Any], d: Dict[str, Any]):
        self.class_ = state.class_
        manager = state.manager
        manager.dispatch.pickle(state, d)

    def __call__(self, state, inst, state_dict):
        state.manager = manager = opt_manager_of_class(self.class_)
        if manager is None:
            raise exc.UnmappedInstanceError(inst, 'Cannot deserialize object of type %r - no mapper() has been configured for this class within the current Python process!' % self.class_)
        elif manager.is_mapped and (not manager.mapper.configured):
            manager.mapper._check_configure()
        if inst is not None:
            manager.setup_instance(inst, state)
        manager.dispatch.unpickle(state, state_dict)