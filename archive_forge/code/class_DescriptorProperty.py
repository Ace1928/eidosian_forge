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
class DescriptorProperty(MapperProperty[_T]):
    """:class:`.MapperProperty` which proxies access to a
    user-defined descriptor."""
    doc: Optional[str] = None
    uses_objects = False
    _links_to_entity = False
    descriptor: DescriptorReference[Any]

    def get_history(self, state: InstanceState[Any], dict_: _InstanceDict, passive: PassiveFlag=PassiveFlag.PASSIVE_OFF) -> History:
        raise NotImplementedError()

    def instrument_class(self, mapper: Mapper[Any]) -> None:
        prop = self

        class _ProxyImpl(attributes.AttributeImpl):
            accepts_scalar_loader = False
            load_on_unexpire = True
            collection = False

            @property
            def uses_objects(self) -> bool:
                return prop.uses_objects

            def __init__(self, key: str):
                self.key = key

            def get_history(self, state: InstanceState[Any], dict_: _InstanceDict, passive: PassiveFlag=PassiveFlag.PASSIVE_OFF) -> History:
                return prop.get_history(state, dict_, passive)
        if self.descriptor is None:
            desc = getattr(mapper.class_, self.key, None)
            if mapper._is_userland_descriptor(self.key, desc):
                self.descriptor = desc
        if self.descriptor is None:

            def fset(obj: Any, value: Any) -> None:
                setattr(obj, self.name, value)

            def fdel(obj: Any) -> None:
                delattr(obj, self.name)

            def fget(obj: Any) -> Any:
                return getattr(obj, self.name)
            self.descriptor = property(fget=fget, fset=fset, fdel=fdel)
        proxy_attr = attributes.create_proxied_attribute(self.descriptor)(self.parent.class_, self.key, self.descriptor, lambda: self._comparator_factory(mapper), doc=self.doc, original_property=self)
        proxy_attr.impl = _ProxyImpl(self.key)
        mapper.class_manager.instrument_attribute(self.key, proxy_attr)