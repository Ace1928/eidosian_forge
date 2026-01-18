from __future__ import annotations
from contextlib import contextmanager
import datetime
import os
import re
import shutil
import sys
from types import ModuleType
from typing import Any
from typing import cast
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from . import revision
from . import write_hooks
from .. import util
from ..runtime import migration
from ..util import compat
from ..util import not_none
@classmethod
def _list_py_dir(cls, scriptdir: ScriptDirectory, path: str) -> List[str]:
    paths = []
    for root, dirs, files in os.walk(path, topdown=True):
        if root.endswith('__pycache__'):
            continue
        for filename in sorted(files):
            paths.append(os.path.join(root, filename))
        if scriptdir.sourceless:
            py_cache_path = os.path.join(root, '__pycache__')
            if os.path.exists(py_cache_path):
                names = {filename.split('.')[0] for filename in files}
                paths.extend((os.path.join(py_cache_path, pyc) for pyc in os.listdir(py_cache_path) if pyc.split('.')[0] not in names))
        if not scriptdir.recursive_version_locations:
            break
        dirs.sort()
    return paths