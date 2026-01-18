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
def _default_path_env_var(self) -> T.Optional[str]:
    env_vars = ['CUDA_PATH'] if self._is_windows() else ['CUDA_PATH', 'CUDA_HOME', 'CUDA_ROOT']
    env_vars = [var for var in env_vars if var in os.environ]
    user_defaults = {os.environ[var] for var in env_vars}
    if len(user_defaults) > 1:
        mlog.warning('Environment variables {} point to conflicting toolkit locations ({}). Toolkit selection might produce unexpected results.'.format(', '.join(env_vars), ', '.join(user_defaults)))
    return env_vars[0] if env_vars else None