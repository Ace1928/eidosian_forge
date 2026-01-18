from __future__ import annotations
from .common import cmake_is_debug
from .. import mlog
from ..mesonlib import Version
from pathlib import Path
import re
import typing as T
def _get_framework_latest_version(path: Path) -> str:
    versions: list[Version] = []
    for each in path.glob('Versions/*'):
        if each.name.lower() == 'current':
            continue
        versions.append(Version(each.name))
    if len(versions) == 0:
        return 'Headers'
    return 'Versions/{}/Headers'.format(sorted(versions)[-1]._s)