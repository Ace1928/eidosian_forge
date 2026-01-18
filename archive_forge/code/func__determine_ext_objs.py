from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, InitVar
from functools import lru_cache
from itertools import chain
from pathlib import Path
import copy
import enum
import json
import os
import pickle
import re
import shlex
import shutil
import typing as T
import hashlib
from .. import build
from .. import dependencies
from .. import programs
from .. import mesonlib
from .. import mlog
from ..compilers import LANGUAGES_USING_LDFLAGS, detect
from ..mesonlib import (
def _determine_ext_objs(self, extobj: 'build.ExtractedObjects', proj_dir_to_build_root: str) -> T.List[str]:
    result: T.List[str] = []
    targetdir = self.get_target_private_dir(extobj.target)
    raw_sources = list(extobj.srclist)
    for gensrc in extobj.genlist:
        for r in gensrc.get_outputs():
            path = self.get_target_generated_dir(extobj.target, gensrc, r)
            dirpart, fnamepart = os.path.split(path)
            raw_sources.append(File(True, dirpart, fnamepart))
    sources: T.List['FileOrString'] = []
    for s in raw_sources:
        if self.environment.is_source(s):
            sources.append(s)
        elif self.environment.is_object(s):
            result.append(s.relative_name())
    if extobj.pch and self.target_uses_pch(extobj.target):
        for lang, pch in extobj.target.pch.items():
            compiler = extobj.target.compilers[lang]
            if compiler.get_argument_syntax() == 'msvc':
                objname = self.get_msvc_pch_objname(lang, pch)
                result.append(os.path.join(proj_dir_to_build_root, targetdir, objname))
    if not sources:
        return result
    if extobj.target.is_unity:
        compsrcs = classify_unity_sources(extobj.target.compilers.values(), sources)
        sources = []
        unity_size = extobj.target.get_option(OptionKey('unity_size'))
        assert isinstance(unity_size, int), 'for mypy'
        for comp, srcs in compsrcs.items():
            if comp.language in LANGS_CANT_UNITY:
                sources += srcs
                continue
            for i in range((len(srcs) + unity_size - 1) // unity_size):
                _src = self.get_unity_source_file(extobj.target, comp.get_default_suffix(), i)
                sources.append(_src)
    for osrc in sources:
        objname = self.object_filename_from_source(extobj.target, osrc, targetdir)
        objpath = os.path.join(proj_dir_to_build_root, objname)
        result.append(objpath)
    return result