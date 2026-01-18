from __future__ import annotations
import importlib
from typing import Generic, TypeVar, Any
from types import ModuleType
class LazyLoad(Generic[_M]):

    def __init__(self, name: str, package: str | None=None) -> None:
        self._lzyname = name
        self._lzypackage = package
        self.__module__: ModuleType | None = None

    def __load__(self) -> _M:
        """Explicitly load the import."""
        if self.__module__ is None:
            self.__module__ = importlib.import_module(self._lzyname, self._lzypackage)
        return self.__module__

    def __reload__(self) -> _M:
        """Explicitly reload the import."""
        try:
            self.__module__ = importlib.reload(self.__module__)
        except Exception as exc:
            try:
                self.__module__ = importlib.import_module(self._lzyname, self._lzypackage)
            except Exception as e:
                raise exc from e
        return self.__module__

    def __repr__(self) -> str:
        """Gives a good representation before import is loaded.
        Uses import's __repr__ after it is loaded.
        """
        if self.__module__ is None:
            if self._lzypackage:
                return f"<Uninitialized module '{self._lzyname}' @ '{self._lzypackage}'>"
            return f"<Uninitialized module '{self._lzyname}'>"
        try:
            return self.__module__.__repr__()
        except AttributeError:
            if self._lzypackage:
                return f"<Initialized module '{self._lzyname}' @ '{self._lzypackage}'>"
            return f"<Initialized module '{self._lzyname}'>"

    def __getattribute__(self, __name: str) -> Any:
        """Proxies attribute access to import (loads the import if not yet loaded)."""
        if __name in {'_lzyname', '_lzypackage', '__module__', '__load__', '__reload__'}:
            return super().__getattribute__(__name)
        if self.__module__ is None:
            self.__module__ = importlib.import_module(self._lzyname, self._lzypackage)
        return getattr(self.__module__, __name)