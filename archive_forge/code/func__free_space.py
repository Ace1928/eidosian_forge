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
def _free_space(self) -> int:
    """Return the number of bytes of free space in the cache directory."""
    return shutil.disk_usage(self._cache_dir)[2]