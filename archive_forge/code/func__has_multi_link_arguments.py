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
def _has_multi_link_arguments(self, args: T.List[str], env: 'Environment', code: str) -> T.Tuple[bool, bool]:
    args = self.linker.fatal_warnings() + args
    args = self.linker_to_compiler_args(args)
    return self.has_arguments(args, env, code, mode=CompileCheckMode.LINK)