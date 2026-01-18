import os
from typing import Optional
import fsspec
from fsspec.archive import AbstractArchiveFileSystem
from fsspec.utils import DEFAULT_BLOCK_SIZE
def _get_dirs(self):
    if self.dir_cache is None:
        f = {**self.file.fs.info(self.file.path), 'name': self.uncompressed_name}
        self.dir_cache = {f['name']: f}