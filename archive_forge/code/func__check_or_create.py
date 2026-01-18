import contextlib
import errno
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import IO, TYPE_CHECKING, ContextManager, Generator, Optional, Tuple
import wandb
from wandb import env, util
from wandb.errors import term
from wandb.sdk.lib.filesystem import files_in
from wandb.sdk.lib.hashutil import B64MD5, ETag, b64_to_hex_id
from wandb.sdk.lib.paths import FilePathStr, StrPath, URIStr
def _check_or_create(self, path: Path, size: int) -> Tuple[FilePathStr, bool, 'Opener']:
    opener = self._cache_opener(path, size)
    hit = path.is_file() and path.stat().st_size == size
    return (FilePathStr(str(path)), hit, opener)