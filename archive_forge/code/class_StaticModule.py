import ast
import importlib
import os
import pathlib
import sys
from glob import iglob
from configparser import ConfigParser
from importlib.machinery import ModuleSpec
from itertools import chain
from typing import (
from pathlib import Path
from types import ModuleType
from distutils.errors import DistutilsOptionError
from .._path import same_path as _same_path
from ..warnings import SetuptoolsWarning
class StaticModule:
    """Proxy to a module object that avoids executing arbitrary code."""

    def __init__(self, name: str, spec: ModuleSpec):
        module = ast.parse(pathlib.Path(spec.origin).read_bytes())
        vars(self).update(locals())
        del self.self

    def _find_assignments(self) -> Iterator[Tuple[ast.AST, ast.AST]]:
        for statement in self.module.body:
            if isinstance(statement, ast.Assign):
                yield from ((target, statement.value) for target in statement.targets)
            elif isinstance(statement, ast.AnnAssign) and statement.value:
                yield (statement.target, statement.value)

    def __getattr__(self, attr):
        """Attempt to load an attribute "statically", via :func:`ast.literal_eval`."""
        try:
            return next((ast.literal_eval(value) for target, value in self._find_assignments() if isinstance(target, ast.Name) and target.id == attr))
        except Exception as e:
            raise AttributeError(f'{self.name} has no attribute {attr}') from e