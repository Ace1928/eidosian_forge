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
class FindLibraryKW(ExtractRequired, ExtractSearchDirs):
    disabler: bool
    has_headers: T.List[str]
    static: bool
    header_args: T.List[str]
    header_dependencies: T.List[dependencies.Dependency]
    header_include_directories: T.List[build.IncludeDirs]
    header_no_builtin_args: bool
    header_prefix: str
    header_required: T.Union[bool, coredata.UserFeatureOption]