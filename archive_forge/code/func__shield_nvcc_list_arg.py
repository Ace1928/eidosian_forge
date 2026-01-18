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
def _shield_nvcc_list_arg(cls, arg: str, listmode: bool=True) -> str:
    """
        Shield an argument against both splitting by NVCC's list-argument
        parse logic, and interpretation by any shell.

        NVCC seems to consider every comma , that is neither escaped by \\ nor inside
        a double-quoted string a split-point. Single-quotes do not provide protection
        against splitting; In fact, after splitting they are \\-escaped. Unfortunately,
        double-quotes don't protect against shell expansion. What follows is a
        complex dance to accommodate everybody.
        """
    SQ = "'"
    DQ = '"'
    CM = ','
    BS = '\\'
    DQSQ = DQ + SQ + DQ
    quotable = set(string.whitespace + '"$`\\')
    if CM not in arg or not listmode:
        if SQ not in arg:
            if set(arg).intersection(quotable):
                return SQ + arg + SQ
            else:
                return arg
        else:
            l = [cls._shield_nvcc_list_arg(s) for s in arg.split(SQ)]
            l = sum([[s, DQSQ] for s in l][:-1], [])
            return ''.join(l)
    else:
        l = ['']
        instring = False
        argit = iter(arg)
        for c in argit:
            if c == CM and (not instring):
                l.append('')
            elif c == DQ:
                l[-1] += c
                instring = not instring
            elif c == BS:
                try:
                    l[-1] += next(argit)
                except StopIteration:
                    break
            else:
                l[-1] += c
        l = [cls._shield_nvcc_list_arg(s, listmode=False) for s in l]
        return '\\,'.join(l)