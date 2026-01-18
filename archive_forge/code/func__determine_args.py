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
def _determine_args(self, kwargs: BaseCompileKW, mode: CompileCheckMode=CompileCheckMode.LINK) -> T.List[str]:
    args: T.List[str] = []
    for i in kwargs['include_directories']:
        for idir in i.to_string_list(self.environment.get_source_dir(), self.environment.get_build_dir()):
            args.extend(self.compiler.get_include_args(idir, False))
    if not kwargs['no_builtin_args']:
        opts = coredata.OptionsView(self.environment.coredata.options, self.subproject)
        args += self.compiler.get_option_compile_args(opts)
        if mode is CompileCheckMode.LINK:
            args.extend(self.compiler.get_option_link_args(opts))
    if kwargs.get('werror', False):
        args.extend(self.compiler.get_werror_args())
    args.extend(kwargs['args'])
    return args