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
def _flatten_object_list(self, target: build.BuildTarget, objects: T.Sequence[T.Union[str, 'File', build.ExtractedObjects]], proj_dir_to_build_root: str) -> T.Tuple[T.List[str], T.List[build.BuildTargetTypes]]:
    obj_list: T.List[str] = []
    deps: T.List[build.BuildTargetTypes] = []
    for obj in objects:
        if isinstance(obj, str):
            o = os.path.join(proj_dir_to_build_root, self.build_to_src, target.get_subdir(), obj)
            obj_list.append(o)
        elif isinstance(obj, mesonlib.File):
            if obj.is_built:
                o = os.path.join(proj_dir_to_build_root, obj.rel_to_builddir(self.build_to_src))
                obj_list.append(o)
            else:
                o = os.path.join(proj_dir_to_build_root, self.build_to_src)
                obj_list.append(obj.rel_to_builddir(o))
        elif isinstance(obj, build.ExtractedObjects):
            if obj.recursive:
                objs, d = self._flatten_object_list(obj.target, obj.objlist, proj_dir_to_build_root)
                obj_list.extend(objs)
                deps.extend(d)
            obj_list.extend(self._determine_ext_objs(obj, proj_dir_to_build_root))
            deps.append(obj.target)
        else:
            raise MesonException('Unknown data type in object list.')
    return (obj_list, deps)