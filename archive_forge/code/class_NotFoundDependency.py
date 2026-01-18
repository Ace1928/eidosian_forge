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
class NotFoundDependency(Dependency):

    def __init__(self, name: str, environment: 'Environment') -> None:
        super().__init__(DependencyTypeName('not-found'), {})
        self.env = environment
        self.name = name
        self.is_found = False

    def get_partial_dependency(self, *, compile_args: bool=False, link_args: bool=False, links: bool=False, includes: bool=False, sources: bool=False) -> 'NotFoundDependency':
        return copy.copy(self)