from __future__ import annotations
from .common import CMakeException, CMakeBuildFile, CMakeConfiguration
import typing as T
from .. import mlog
from pathlib import Path
import json
import re
def _parse_cmakeFiles(self, data: T.Dict[str, T.Any]) -> None:
    assert 'inputs' in data
    assert 'paths' in data
    src_dir = Path(data['paths']['source'])
    for i in data['inputs']:
        path = Path(i['path'])
        path = path if path.is_absolute() else src_dir / path
        self.cmake_sources += [CMakeBuildFile(path, i.get('isCMake', False), i.get('isGenerated', False))]