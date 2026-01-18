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
class _ModuleMarker(ClsRegistryToken):
    """Refers to a module name within
    _decl_class_registry.

    """
    __slots__ = ('parent', 'name', 'contents', 'mod_ns', 'path', '__weakref__')
    parent: Optional[_ModuleMarker]
    contents: Dict[str, Union[_ModuleMarker, _MultipleClassMarker]]
    mod_ns: _ModNS
    path: List[str]

    def __init__(self, name: str, parent: Optional[_ModuleMarker]):
        self.parent = parent
        self.name = name
        self.contents = {}
        self.mod_ns = _ModNS(self)
        if self.parent:
            self.path = self.parent.path + [self.name]
        else:
            self.path = []
        _registries.add(self)

    def __contains__(self, name: str) -> bool:
        return name in self.contents

    def __getitem__(self, name: str) -> ClsRegistryToken:
        return self.contents[name]

    def _remove_item(self, name: str) -> None:
        self.contents.pop(name, None)
        if not self.contents and self.parent is not None:
            self.parent._remove_item(self.name)
            _registries.discard(self)

    def resolve_attr(self, key: str) -> Union[_ModNS, Type[Any]]:
        return self.mod_ns.__getattr__(key)

    def get_module(self, name: str) -> _ModuleMarker:
        if name not in self.contents:
            marker = _ModuleMarker(name, self)
            self.contents[name] = marker
        else:
            marker = cast(_ModuleMarker, self.contents[name])
        return marker

    def add_class(self, name: str, cls: Type[Any]) -> None:
        if name in self.contents:
            existing = cast(_MultipleClassMarker, self.contents[name])
            try:
                existing.add_item(cls)
            except AttributeError as ae:
                if not isinstance(existing, _MultipleClassMarker):
                    raise exc.InvalidRequestError(f'name "{name}" matches both a class name and a module name') from ae
                else:
                    raise
        else:
            existing = self.contents[name] = _MultipleClassMarker([cls], on_remove=lambda: self._remove_item(name))

    def remove_class(self, name: str, cls: Type[Any]) -> None:
        if name in self.contents:
            existing = cast(_MultipleClassMarker, self.contents[name])
            existing.remove_item(cls)