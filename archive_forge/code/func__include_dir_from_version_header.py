from __future__ import annotations
import re
import dataclasses
import functools
import typing as T
from pathlib import Path
from .. import mlog
from .. import mesonlib
from .base import DependencyException, SystemDependency
from .detect import packages
from .pkgconfig import PkgConfigDependency
from .misc import threads_factory
def _include_dir_from_version_header(self, hfile: Path) -> BoostIncludeDir:
    assert hfile.exists()
    raw = hfile.read_text(encoding='utf-8')
    m = re.search('#define\\s+BOOST_VERSION\\s+([0-9]+)', raw)
    if not m:
        mlog.debug(f'Failed to extract version information from {hfile}')
        return BoostIncludeDir(hfile.parents[1], 0)
    return BoostIncludeDir(hfile.parents[1], int(m.group(1)))