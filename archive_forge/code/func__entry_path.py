import contextlib
import hashlib
import logging
import os
from types import TracebackType
from typing import Dict, Generator, Optional, Set, Type, Union
from pip._internal.models.link import Link
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.temp_dir import TempDirectory
def _entry_path(self, key: TrackerId) -> str:
    hashed = hashlib.sha224(key.encode()).hexdigest()
    return os.path.join(self._root, hashed)