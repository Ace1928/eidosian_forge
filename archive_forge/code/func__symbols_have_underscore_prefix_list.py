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
def _symbols_have_underscore_prefix_list(self, env: 'Environment') -> T.Optional[bool]:
    """
        Check if symbols have underscore prefix by consulting a hardcoded
        list of cases where we know the results.
        Return if functions have underscore prefix or None if unknown.
        """
    m = env.machines[self.for_machine]
    if m.is_darwin():
        return True
    if m.is_windows() or m.is_cygwin():
        return m.cpu_family == 'x86'
    return None