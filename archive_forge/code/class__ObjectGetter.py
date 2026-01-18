from typing import (Any, Callable, Dict, Generic, Iterator, List, Optional,
from .pyutils import get_named_object
class _ObjectGetter(Generic[T]):
    """Maintain a reference to an object, and return the object on request.

    This is used by Registry to make plain objects function similarly
    to lazily imported objects.

    Objects can be any sort of python object (class, function, module,
    instance, etc)
    """
    __slots__ = ['_obj']
    _obj: T

    def __init__(self, obj):
        self._obj = obj

    def get_module(self) -> str:
        """Get the module the object was loaded from."""
        return self._obj.__module__

    def get_obj(self) -> T:
        """Get the object that was saved at creation time"""
        return self._obj