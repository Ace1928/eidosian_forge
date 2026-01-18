from __future__ import annotations
import os
import abc
import contextlib
import multiprocessing
from pathlib import Path
from .types import AppEnv
from lazyops.libs.proxyobj import proxied
from typing import Optional, Dict, Any, List, Union, Type, Callable, TypeVar, Literal, overload, TYPE_CHECKING
@proxied
class ApplicationContextManager(abc.ABC):
    """
    The context manager for the app context
    """

    def __init__(self, *args, **kwargs):
        """
        Creates the context manager
        """
        self.ctxs: Dict[str, ApplicationContext] = {}
        self._global_ctx = None

    @property
    def global_ctx(self):
        """
        Returns the global context
        """
        if self._global_ctx is None:
            from ..state import GlobalContext
            self._global_ctx = GlobalContext
        return self._global_ctx

    def init_ctx(self, module_name: str, *args, **kwargs) -> ApplicationContext:
        """
        Initializes the app context
        """
        if module_name not in self.ctxs:
            self.ctxs[module_name] = ApplicationContext(module_name, *args, global_ctx=self.global_ctx, **kwargs)
        return self.ctxs[module_name]

    def get_ctx(self, module_name: str, *args, **kwargs) -> ApplicationContext:
        """
        Retrieves the app context
        """
        if module_name not in self.ctxs:
            return self.init_ctx(module_name, *args, **kwargs)
        return self.ctxs[module_name]

    def __getitem__(self, module_name: str) -> ApplicationContext:
        """
        Retrieves the app context
        """
        return self.get_ctx(module_name)