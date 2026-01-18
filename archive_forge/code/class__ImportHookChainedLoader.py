import sys
import threading
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Union
class _ImportHookChainedLoader:

    def __init__(self, loader: Any) -> None:
        self.loader = loader
        if hasattr(loader, 'load_module'):
            self.load_module = self._load_module
        if hasattr(loader, 'create_module'):
            self.create_module = self._create_module
        if hasattr(loader, 'exec_module'):
            self.exec_module = self._exec_module

    def _set_loader(self, module: Any) -> None:

        class UNDEFINED:
            pass
        if getattr(module, '__loader__', UNDEFINED) in (None, self):
            try:
                module.__loader__ = self.loader
            except AttributeError:
                pass
        if getattr(module, '__spec__', None) is not None and getattr(module.__spec__, 'loader', None) is self:
            module.__spec__.loader = self.loader

    def _load_module(self, fullname: str) -> Any:
        module = self.loader.load_module(fullname)
        self._set_loader(module)
        notify_module_loaded(module)
        return module

    def _create_module(self, spec: Any) -> Any:
        return self.loader.create_module(spec)

    def _exec_module(self, module: Any) -> None:
        self._set_loader(module)
        self.loader.exec_module(module)
        notify_module_loaded(module)