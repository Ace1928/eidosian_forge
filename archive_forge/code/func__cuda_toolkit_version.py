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
def _cuda_toolkit_version(self, path: str) -> str:
    version = self._read_toolkit_version_txt(path)
    if version:
        return version
    version = self._read_cuda_runtime_api_version(path)
    if version:
        return version
    mlog.debug('Falling back to extracting version from path')
    path_version_regex = self.path_version_win_regex if self._is_windows() else self.path_version_nix_regex
    try:
        m = path_version_regex.match(os.path.basename(path))
        if m:
            return m.group(1)
        else:
            mlog.warning(f'Could not detect CUDA Toolkit version for {path}')
    except Exception as e:
        mlog.warning(f'Could not detect CUDA Toolkit version for {path}: {e!s}')
    return '0.0'