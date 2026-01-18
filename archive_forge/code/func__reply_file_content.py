from __future__ import annotations
from .common import CMakeException, CMakeBuildFile, CMakeConfiguration
import typing as T
from .. import mlog
from pathlib import Path
import json
import re
def _reply_file_content(self, filename: Path) -> T.Dict[str, T.Any]:
    real_path = self.reply_dir / filename
    if not real_path.exists():
        raise CMakeException(f'File "{real_path}" does not exist')
    data = json.loads(real_path.read_text(encoding='utf-8'))
    assert isinstance(data, dict)
    for i in data.keys():
        assert isinstance(i, str)
    return data