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
def _get_file_from_list(env: Environment, paths: T.List[Path]) -> T.Optional[Path]:
    """
        We just check whether the library exists. We can't do a link check
        because the library might have unresolved symbols that require other
        libraries. On macOS we check if the library matches our target
        architecture.
        """
    for p in paths:
        if p.is_file():
            if env.machines.host.is_darwin() and env.machines.build.is_darwin():
                archs = mesonlib.darwin_get_object_archs(str(p))
                if not archs or env.machines.host.cpu_family not in archs:
                    mlog.debug(f'Rejected {p}, supports {archs} but need {env.machines.host.cpu_family}')
                    continue
            return p
    return None