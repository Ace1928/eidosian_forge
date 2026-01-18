from __future__ import annotations
import enum
import os.path
import string
import typing as T
from .. import coredata
from .. import mlog
from ..mesonlib import (
from .compilers import Compiler
@classmethod
def _merge_flags(cls, flags: T.List[str]) -> T.List[str]:
    """
        The flags to NVCC gets exceedingly verbose and unreadable when too many of them
        are shielded with -Xcompiler. Merge consecutive -Xcompiler-wrapped arguments
        into one.
        """
    if len(flags) <= 1:
        return flags
    flagit = iter(flags)
    xflags = []

    def is_xcompiler_flag_isolated(flag: str) -> bool:
        return flag == '-Xcompiler'

    def is_xcompiler_flag_glued(flag: str) -> bool:
        return flag.startswith('-Xcompiler=')

    def is_xcompiler_flag(flag: str) -> bool:
        return is_xcompiler_flag_isolated(flag) or is_xcompiler_flag_glued(flag)

    def get_xcompiler_val(flag: str, flagit: T.Iterator[str]) -> str:
        if is_xcompiler_flag_glued(flag):
            return flag[len('-Xcompiler='):]
        else:
            try:
                return next(flagit)
            except StopIteration:
                return ''
    ingroup = False
    for flag in flagit:
        if not is_xcompiler_flag(flag):
            ingroup = False
            xflags.append(flag)
        elif ingroup:
            xflags[-1] += ','
            xflags[-1] += get_xcompiler_val(flag, flagit)
        elif is_xcompiler_flag_isolated(flag):
            ingroup = True
            xflags.append(flag)
            xflags.append(get_xcompiler_val(flag, flagit))
        elif is_xcompiler_flag_glued(flag):
            ingroup = True
            xflags.append(flag)
        else:
            raise ValueError('-Xcompiler flag merging failed, unknown argument form!')
    return xflags