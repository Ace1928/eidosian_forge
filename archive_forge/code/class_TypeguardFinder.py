from __future__ import annotations
import ast
import sys
import types
from collections.abc import Callable, Iterable
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, SourceFileLoader
from importlib.util import cache_from_source, decode_source
from inspect import isclass
from os import PathLike
from types import CodeType, ModuleType, TracebackType
from typing import Sequence, TypeVar
from unittest.mock import patch
from ._config import global_config
from ._transformer import TypeguardTransformer
class TypeguardFinder(MetaPathFinder):
    """
    Wraps another path finder and instruments the module with
    :func:`@typechecked <typeguard.typechecked>` if :meth:`should_instrument` returns
    ``True``.

    Should not be used directly, but rather via :func:`~.install_import_hook`.

    .. versionadded:: 2.6
    """

    def __init__(self, packages: list[str] | None, original_pathfinder: MetaPathFinder):
        self.packages = packages
        self._original_pathfinder = original_pathfinder

    def find_spec(self, fullname: str, path: Sequence[str] | None, target: types.ModuleType | None=None) -> ModuleSpec | None:
        if self.should_instrument(fullname):
            spec = self._original_pathfinder.find_spec(fullname, path, target)
            if spec is not None and isinstance(spec.loader, SourceFileLoader):
                spec.loader = TypeguardLoader(spec.loader.name, spec.loader.path)
                return spec
        return None

    def should_instrument(self, module_name: str) -> bool:
        """
        Determine whether the module with the given name should be instrumented.

        :param module_name: full name of the module that is about to be imported (e.g.
            ``xyz.abc``)

        """
        if self.packages is None:
            return True
        for package in self.packages:
            if module_name == package or module_name.startswith(package + '.'):
                return True
        return False