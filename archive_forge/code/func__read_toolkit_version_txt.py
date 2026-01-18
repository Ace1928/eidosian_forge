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
def _read_toolkit_version_txt(self, path: str) -> T.Optional[str]:
    version_file_path = os.path.join(path, 'version.txt')
    try:
        with open(version_file_path, encoding='utf-8') as version_file:
            version_str = version_file.readline()
            m = self.toolkit_version_regex.match(version_str)
            if m:
                return self._strip_patch_version(m.group(1))
    except Exception as e:
        mlog.debug(f"Could not read CUDA Toolkit's version file {version_file_path}: {e!s}")
    return None