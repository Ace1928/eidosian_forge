import abc
from collections import defaultdict
import collections.abc
from contextlib import contextmanager
import os
from pathlib import (  # type: ignore
import shutil
import sys
from typing import (
from urllib.parse import urlparse
from warnings import warn
from cloudpathlib.enums import FileCacheMode
from . import anypath
from .exceptions import (
class CloudPathMeta(abc.ABCMeta):

    @overload
    def __call__(cls: Type[T], cloud_path: CloudPathT, *args: Any, **kwargs: Any) -> CloudPathT:
        ...

    @overload
    def __call__(cls: Type[T], cloud_path: Union[str, 'CloudPath'], *args: Any, **kwargs: Any) -> T:
        ...

    def __call__(cls: Type[T], cloud_path: Union[str, CloudPathT], *args: Any, **kwargs: Any) -> Union[T, 'CloudPath', CloudPathT]:
        if not issubclass(cls, CloudPath):
            raise TypeError(f'Only subclasses of {CloudPath.__name__} can be instantiated from its meta class.')
        if cls is CloudPath:
            for implementation in implementation_registry.values():
                path_class = implementation._path_class
                if path_class is not None and path_class.is_valid_cloudpath(cloud_path, raise_on_error=False):
                    new_obj = object.__new__(path_class)
                    path_class.__init__(new_obj, cloud_path, *args, **kwargs)
                    return new_obj
            valid_prefixes = [impl._path_class.cloud_prefix for impl in implementation_registry.values() if impl._path_class is not None]
            raise InvalidPrefixError(f'Path {cloud_path} does not begin with a known prefix {valid_prefixes}.')
        new_obj = object.__new__(cls)
        cls.__init__(new_obj, cloud_path, *args, **kwargs)
        return new_obj

    def __init__(cls, name: str, bases: Tuple[type, ...], dic: Dict[str, Any]) -> None:
        for attr in dir(cls):
            if not attr.startswith('_') and hasattr(Path, attr) and getattr(getattr(Path, attr), '__doc__', None):
                docstring = getattr(Path, attr).__doc__ + ' _(Docstring copied from pathlib.Path)_'
                getattr(cls, attr).__doc__ = docstring
                if isinstance(getattr(cls, attr), property):
                    getattr(cls, attr).fget.__doc__ = docstring