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
def _detect_cuda_path_and_version(self) -> TV_ResultTuple:
    self.env_var = self._default_path_env_var()
    mlog.debug('Default path env var:', mlog.bold(self.env_var))
    version_reqs = self.version_reqs
    if self.language == 'cuda':
        nvcc_version = self._strip_patch_version(self.get_compiler().version)
        mlog.debug('nvcc version:', mlog.bold(nvcc_version))
        if version_reqs:
            found_some, not_found, found = mesonlib.version_compare_many(nvcc_version, version_reqs)
            if not_found:
                msg = f'The current nvcc version {nvcc_version} does not satisfy the specified CUDA Toolkit version requirements {version_reqs}.'
                return self._report_dependency_error(msg, (None, None, False))
        version_reqs = [f'={nvcc_version}']
    else:
        nvcc_version = None
    paths = [(path, self._cuda_toolkit_version(path), default) for path, default in self._cuda_paths()]
    if version_reqs:
        return self._find_matching_toolkit(paths, version_reqs, nvcc_version)
    defaults = [(path, version) for path, version, default in paths if default]
    if defaults:
        return (defaults[0][0], defaults[0][1], True)
    platform_msg = 'set the CUDA_PATH environment variable' if self._is_windows() else "set the CUDA_PATH environment variable/create the '/usr/local/cuda' symbolic link"
    msg = f"Please specify the desired CUDA Toolkit version (e.g. dependency('cuda', version : '>=10.1')) or {platform_msg} to point to the location of your desired version."
    return self._report_dependency_error(msg, (None, None, False))