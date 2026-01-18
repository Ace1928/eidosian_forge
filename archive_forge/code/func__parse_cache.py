from __future__ import annotations
from .common import CMakeException, CMakeBuildFile, CMakeConfiguration
import typing as T
from .. import mlog
from pathlib import Path
import json
import re
def _parse_cache(self, data: T.Dict[str, T.Any]) -> None:
    assert 'entries' in data
    for e in data['entries']:
        if e['name'] == 'CMAKE_PROJECT_VERSION':
            self.project_version = e['value']