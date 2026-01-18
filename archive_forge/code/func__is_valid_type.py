from __future__ import annotations
import os
import json
import re
import sys
import shutil
import typing as T
from collections import defaultdict
from pathlib import Path
from . import mlog
from . import mesonlib
from .mesonlib import MesonException, RealPathAction, join_args, setup_vsenv
from mesonbuild.environment import detect_ninja
from mesonbuild.coredata import UserArrayOption
from mesonbuild import build
@staticmethod
def _is_valid_type(type: str) -> bool:
    allowed_types = {'executable', 'static_library', 'shared_library', 'shared_module', 'custom', 'alias', 'run', 'jar'}
    return type in allowed_types