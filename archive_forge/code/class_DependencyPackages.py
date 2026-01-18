from __future__ import annotations
import collections, functools, importlib
import typing as T
from .base import ExternalDependency, DependencyException, DependencyMethods, NotFoundDependency
from ..mesonlib import listify, MachineChoice, PerMachine
from .. import mlog
class DependencyPackages(collections.UserDict):
    data: T.Dict[str, PackageTypes]
    defaults: T.Dict[str, str] = {}

    def __missing__(self, key: str) -> PackageTypes:
        if key in self.defaults:
            modn = self.defaults[key]
            importlib.import_module(f'mesonbuild.dependencies.{modn}')
            return self.data[key]
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return key in self.defaults or key in self.data