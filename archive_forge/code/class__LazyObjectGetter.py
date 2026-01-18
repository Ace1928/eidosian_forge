from typing import (Any, Callable, Dict, Generic, Iterator, List, Optional,
from .pyutils import get_named_object
class _LazyObjectGetter(_ObjectGetter[T]):
    """Keep a record of a possible object.

    When requested, load and return it.
    """
    __slots__ = ['_module_name', '_member_name', '_imported']

    def __init__(self, module_name, member_name):
        self._module_name = module_name
        self._member_name = member_name
        self._imported = False
        super().__init__(None)

    def get_module(self):
        """Get the module the referenced object will be loaded from.
        """
        return self._module_name

    def get_obj(self) -> T:
        """Get the referenced object.

        Upon first request, the object will be imported. Future requests will
        return the imported object.
        """
        if not self._imported:
            self._obj = get_named_object(self._module_name, self._member_name)
            self._imported = True
        return super().get_obj()

    def __repr__(self):
        return '<{}.{} object at {:x}, module={!r} attribute={!r} imported={!r}>'.format(self.__class__.__module__, self.__class__.__name__, id(self), self._module_name, self._member_name, self._imported)