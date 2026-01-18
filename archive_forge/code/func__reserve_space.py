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
def _reserve_space(self, size: int) -> None:
    """If a `size` write would exceed disk space, remove cached items to make space.

        Raises:
            OSError: If there is not enough space to write `size` bytes, even after
                removing cached items.
        """
    if size <= self._free_space():
        return
    term.termwarn('Cache size exceeded. Attempting to reclaim space...')
    self.cleanup(target_fraction=0.5)
    if size <= self._free_space():
        return
    self.cleanup(target_size=0)
    if size > self._free_space():
        raise OSError(errno.ENOSPC, f'Insufficient free space in {self._cache_dir}')