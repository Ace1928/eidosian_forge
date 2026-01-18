from __future__ import annotations
import collections
import enum
import functools
import os
import itertools
import typing as T
from .. import build
from .. import coredata
from .. import dependencies
from .. import mesonlib
from .. import mlog
from ..compilers import SUFFIX_TO_LANG
from ..compilers.compilers import CompileCheckMode
from ..interpreterbase import (ObjectHolder, noPosargs, noKwargs,
from ..interpreterbase.decorators import ContainerTypeInfo, typed_kwargs, KwargInfo, typed_pos_args
from ..mesonlib import OptionKey
from .interpreterobjects import (extract_required_kwarg, extract_search_dirs)
from .type_checking import REQUIRED_KW, in_set_validator, NoneType
def _determine_dependencies(self, deps: T.List['dependencies.Dependency'], compile_only: bool=False, endl: str=':') -> T.Tuple[T.List['dependencies.Dependency'], str]:
    deps = dependencies.get_leaf_external_dependencies(deps)
    return (deps, self._dep_msg(deps, compile_only, endl))