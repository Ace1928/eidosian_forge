from __future__ import annotations
import copy
import os
import collections
import itertools
import typing as T
from enum import Enum
from .. import mlog, mesonlib
from ..compilers import clib_langs
from ..mesonlib import LibType, MachineChoice, MesonException, HoldableObject, OptionKey
from ..mesonlib import version_compare_many
class MissingCompiler(_MissingCompilerBase):
    """Represent a None Compiler - when no tool chain is found.
    replacing AttributeError with DependencyException"""
    if T.TYPE_CHECKING:

        def __init__(self) -> None:
            pass

        def get_optimization_args(self, optimization_level: str) -> T.List[str]:
            return []

        def get_output_args(self, outputname: str) -> T.List[str]:
            return []

        def sanity_check(self, work_dir: str, environment: 'Environment') -> None:
            return None

    def __getattr__(self, item: str) -> T.Any:
        if item.startswith('__'):
            raise AttributeError()
        raise DependencyException('no toolchain found')

    def __bool__(self) -> bool:
        return False