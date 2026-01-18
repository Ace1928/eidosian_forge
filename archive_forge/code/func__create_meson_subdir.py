from __future__ import annotations
import dataclasses
import glob
import importlib
import itertools
import json
import os
import shutil
import collections
import typing as T
from . import builder
from . import version
from ..mesonlib import MesonException, Popen_safe, OptionKey
from .. import coredata
def _create_meson_subdir(cargo: Manifest, build: builder.Builder) -> T.List[mparser.BaseNode]:
    return [build.assign(build.array([]), _extra_args_varname()), build.assign(build.array([]), _extra_deps_varname()), build.assign(build.function('import', [build.string('fs')]), 'fs'), build.if_(build.method('is_dir', build.identifier('fs'), [build.string('meson')]), build.block([build.function('subdir', [build.string('meson')])]))]