from __future__ import annotations
import glob
import re
import os
import typing as T
from pathlib import Path
from .. import mesonlib
from .. import mlog
from ..environment import detect_cpu_family
from .base import DependencyException, SystemDependency
from .detect import packages
def _cuda_paths_nix(self) -> T.List[T.Tuple[str, bool]]:
    pattern = '/usr/local/cuda-*' if self.env_var else '/usr/local/cuda*'
    return [(path, os.path.basename(path) == 'cuda') for path in glob.iglob(pattern)]