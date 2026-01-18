import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
def _ensure_no_accidental_inclusion(self, detected: List[str], kind: str):
    if len(detected) > 1:
        from inspect import cleandoc
        from setuptools.errors import PackageDiscoveryError
        msg = f'Multiple top-level {kind} discovered in a flat-layout: {detected}.\n\n            To avoid accidental inclusion of unwanted files or directories,\n            setuptools will not proceed with this build.\n\n            If you are trying to create a single distribution with multiple {kind}\n            on purpose, you should not rely on automatic discovery.\n            Instead, consider the following options:\n\n            1. set up custom discovery (`find` directive with `include` or `exclude`)\n            2. use a `src-layout`\n            3. explicitly set `py_modules` or `packages` with a list of names\n\n            To find more information, look for "package discovery" on setuptools docs.\n            '
        raise PackageDiscoveryError(cleandoc(msg))