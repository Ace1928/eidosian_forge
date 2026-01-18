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
@classmethod
def _process_include_type_kw(cls, kwargs: T.Dict[str, T.Any]) -> str:
    if 'include_type' not in kwargs:
        return 'preserve'
    if not isinstance(kwargs['include_type'], str):
        raise DependencyException('The include_type kwarg must be a string type')
    if kwargs['include_type'] not in ['preserve', 'system', 'non-system']:
        raise DependencyException("include_type may only be one of ['preserve', 'system', 'non-system']")
    return kwargs['include_type']