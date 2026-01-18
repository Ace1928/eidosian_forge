from __future__ import annotations
from dataclasses import is_dataclass
import inspect
import itertools
import operator
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import util as orm_util
from .base import _DeclarativeMapped
from .base import LoaderCallableStatus
from .base import Mapped
from .base import PassiveFlag
from .base import SQLORMOperations
from .interfaces import _AttributeOptions
from .interfaces import _IntrospectsAnnotations
from .interfaces import _MapsColumns
from .interfaces import MapperProperty
from .interfaces import PropComparator
from .util import _none_set
from .util import de_stringify_annotation
from .. import event
from .. import exc as sa_exc
from .. import schema
from .. import sql
from .. import util
from ..sql import expression
from ..sql import operators
from ..sql.elements import BindParameter
from ..util.typing import is_fwd_ref
from ..util.typing import is_pep593
from ..util.typing import typing_get_args
class ConcreteInheritedProperty(DescriptorProperty[_T]):
    """A 'do nothing' :class:`.MapperProperty` that disables
    an attribute on a concrete subclass that is only present
    on the inherited mapper, not the concrete classes' mapper.

    Cases where this occurs include:

    * When the superclass mapper is mapped against a
      "polymorphic union", which includes all attributes from
      all subclasses.
    * When a relationship() is configured on an inherited mapper,
      but not on the subclass mapper.  Concrete mappers require
      that relationship() is configured explicitly on each
      subclass.

    """

    def _comparator_factory(self, mapper: Mapper[Any]) -> Type[PropComparator[_T]]:
        comparator_callable = None
        for m in self.parent.iterate_to_root():
            p = m._props[self.key]
            if getattr(p, 'comparator_factory', None) is not None:
                comparator_callable = p.comparator_factory
                break
        assert comparator_callable is not None
        return comparator_callable(p, mapper)

    def __init__(self) -> None:
        super().__init__()

        def warn() -> NoReturn:
            raise AttributeError('Concrete %s does not implement attribute %r at the instance level.  Add this property explicitly to %s.' % (self.parent, self.key, self.parent))

        class NoninheritedConcreteProp:

            def __set__(s: Any, obj: Any, value: Any) -> NoReturn:
                warn()

            def __delete__(s: Any, obj: Any) -> NoReturn:
                warn()

            def __get__(s: Any, obj: Any, owner: Any) -> Any:
                if obj is None:
                    return self.descriptor
                warn()
        self.descriptor = NoninheritedConcreteProp()