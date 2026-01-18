from __future__ import annotations
from dataclasses import dataclass
import subprocess
import typing as T
from enum import Enum
from . import mesonlib
from .mesonlib import EnvironmentException, HoldableObject
from . import mlog
from pathlib import Path
class CMakeVariables:

    def __init__(self, variables: T.Optional[T.Dict[str, T.Any]]=None) -> None:
        variables = variables or {}
        self.variables: T.Dict[str, T.List[str]] = {}
        for key, value in variables.items():
            value = mesonlib.listify(value)
            for i in value:
                if not isinstance(i, str):
                    raise EnvironmentException(f"Value '{i}' of CMake variable '{key}' defined in a machine file is a {type(i).__name__} and not a str")
            self.variables[key] = value

    def get_variables(self) -> T.Dict[str, T.List[str]]:
        return self.variables