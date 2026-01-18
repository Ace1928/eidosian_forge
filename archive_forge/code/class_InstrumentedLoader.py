from typing import Any, Callable, cast, List, Optional
from types import ModuleType
from importlib.machinery import ModuleSpec
from importlib.abc import Loader
from contextlib import contextmanager
import importlib
from importlib import abc
import sys
class InstrumentedLoader(abc.Loader):
    """A module loader used to hook the python import statement."""

    def __init__(self, loader: Any, wrap_module: Callable[[ModuleType], Optional[ModuleType]], after_exec: Callable[[ModuleType], None]):
        """A module loader that uses an existing module loader and intercepts
        the execution of a module.

        Use `InstrumentedFinder` to instrument modules with instances of this
        class.

        Args:
            loader: The original module loader to wrap.
            module_name: The fully qualified module name to instrument e.g.
                `'pkg.submodule'`.  Submodules of this are also instrumented.
            wrap_module: A callback function that takes a module object before
                it is run and either modifies or replaces it before it is run.
                The module returned by this function will be executed.  If None
                is returned the module is not executed and may be executed
                later.
            after_exec: A callback function that is called with the return value
                of `wrap_module` after that module was executed if `wrap_module`
                didn't return None.
        """
        self.loader = loader
        self.wrap_module = wrap_module
        self.after_exec = after_exec

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        return self.loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        wrapped_module = self.wrap_module(module)
        if wrapped_module is not None:
            self.loader.exec_module(module)
            self.after_exec(module)