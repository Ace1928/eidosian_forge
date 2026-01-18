from __future__ import annotations
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NoReturn
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import interfaces
from .descriptor_props import SynonymProperty
from .properties import ColumnProperty
from .util import class_mapper
from .. import exc
from .. import inspection
from .. import util
from ..sql.schema import _get_table_key
from ..util.typing import CallableReference
class _GetColumns:
    __slots__ = ('cls',)
    cls: Type[Any]

    def __init__(self, cls: Type[Any]):
        self.cls = cls

    def __getattr__(self, key: str) -> Any:
        mp = class_mapper(self.cls, configure=False)
        if mp:
            if key not in mp.all_orm_descriptors:
                raise AttributeError('Class %r does not have a mapped column named %r' % (self.cls, key))
            desc = mp.all_orm_descriptors[key]
            if desc.extension_type is interfaces.NotExtension.NOT_EXTENSION:
                assert isinstance(desc, attributes.QueryableAttribute)
                prop = desc.property
                if isinstance(prop, SynonymProperty):
                    key = prop.name
                elif not isinstance(prop, ColumnProperty):
                    raise exc.InvalidRequestError('Property %r is not an instance of ColumnProperty (i.e. does not correspond directly to a Column).' % key)
        return getattr(self.cls, key)