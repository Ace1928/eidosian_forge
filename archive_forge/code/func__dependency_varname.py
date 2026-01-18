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
def _dependency_varname(package_name: str) -> str:
    return f'{fixup_meson_varname(package_name)}_dep'