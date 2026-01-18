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
class _ModNS:
    __slots__ = ('__parent',)
    __parent: _ModuleMarker

    def __init__(self, parent: _ModuleMarker):
        self.__parent = parent

    def __getattr__(self, key: str) -> Union[_ModNS, Type[Any]]:
        try:
            value = self.__parent.contents[key]
        except KeyError:
            pass
        else:
            if value is not None:
                if isinstance(value, _ModuleMarker):
                    return value.mod_ns
                else:
                    assert isinstance(value, _MultipleClassMarker)
                    return value.attempt_get(self.__parent.path, key)
        raise NameError('Module %r has no mapped classes registered under the name %r' % (self.__parent.name, key))