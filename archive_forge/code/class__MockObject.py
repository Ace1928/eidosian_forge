import contextlib
import os
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import MethodType, ModuleType
from typing import Any, Generator, Iterator, List, Optional, Sequence, Tuple, Union
from sphinx.util import logging
from sphinx.util.inspect import isboundmethod, safe_getattr
class _MockObject:
    """Used by autodoc_mock_imports."""
    __display_name__ = '_MockObject'
    __name__ = ''
    __sphinx_mock__ = True
    __sphinx_decorator_args__: Tuple[Any, ...] = ()

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 3 and isinstance(args[1], tuple):
            superclass = args[1][-1].__class__
            if superclass is cls:
                return _make_subclass(args[0], superclass.__display_name__, superclass=superclass, attributes=args[2])
        return super().__new__(cls)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.__qualname__ = self.__name__

    def __len__(self) -> int:
        return 0

    def __contains__(self, key: str) -> bool:
        return False

    def __iter__(self) -> Iterator:
        return iter([])

    def __mro_entries__(self, bases: Tuple) -> Tuple:
        return (self.__class__,)

    def __getitem__(self, key: Any) -> '_MockObject':
        return _make_subclass(str(key), self.__display_name__, self.__class__)()

    def __getattr__(self, key: str) -> '_MockObject':
        return _make_subclass(key, self.__display_name__, self.__class__)()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        call = self.__class__()
        call.__sphinx_decorator_args__ = args
        return call

    def __repr__(self) -> str:
        return self.__display_name__