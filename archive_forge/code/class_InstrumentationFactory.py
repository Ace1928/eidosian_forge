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
class InstrumentationFactory(EventTarget):
    """Factory for new ClassManager instances."""
    dispatch: dispatcher[InstrumentationFactory]

    def create_manager_for_cls(self, class_: Type[_O]) -> ClassManager[_O]:
        assert class_ is not None
        assert opt_manager_of_class(class_) is None
        manager, factory = self._locate_extended_factory(class_)
        if factory is None:
            factory = ClassManager
            manager = ClassManager(class_)
        else:
            assert manager is not None
        self._check_conflicts(class_, factory)
        manager.factory = factory
        return manager

    def _locate_extended_factory(self, class_: Type[_O]) -> Tuple[Optional[ClassManager[_O]], Optional[_ManagerFactory]]:
        """Overridden by a subclass to do an extended lookup."""
        return (None, None)

    def _check_conflicts(self, class_: Type[_O], factory: Callable[[Type[_O]], ClassManager[_O]]) -> None:
        """Overridden by a subclass to test for conflicting factories."""

    def unregister(self, class_: Type[_O]) -> None:
        manager = manager_of_class(class_)
        manager.unregister()
        self.dispatch.class_uninstrument(class_)