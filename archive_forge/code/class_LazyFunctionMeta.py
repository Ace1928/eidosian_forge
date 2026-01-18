from importlib import import_module
from typing import Callable
from functools import lru_cache, wraps
class LazyFunctionMeta(type):

    @property
    def __doc__(self):
        docstring = _get_function().__doc__
        docstring += f"\n\nNote: this is a {self.__class__.__name__} wrapper of '{module}.{name}'"
        return docstring