from __future__ import annotations
import collections
import functools
import glob
import itertools
import os
import re
import subprocess
import copy
import typing as T
from pathlib import Path
from ... import arglist
from ... import mesonlib
from ... import mlog
from ...linkers.linkers import GnuLikeDynamicLinkerMixin, SolarisDynamicLinker, CompCertDynamicLinker
from ...mesonlib import LibType, OptionKey
from .. import compilers
from ..compilers import CompileCheckMode
from .visualstudio import VisualStudioLikeCompiler
@staticmethod
def _concatenate_string_literals(s: str) -> str:
    pattern = re.compile('(?P<pre>.*([^\\\\]")|^")(?P<str1>([^\\\\"]|\\\\.)*)"\\s+"(?P<str2>([^\\\\"]|\\\\.)*)(?P<post>".*)')
    ret = s
    m = pattern.match(ret)
    while m:
        ret = ''.join(m.group('pre', 'str1', 'str2', 'post'))
        m = pattern.match(ret)
    return ret