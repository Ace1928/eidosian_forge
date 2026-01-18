import os
from typing import Optional
import fsspec
from fsspec.archive import AbstractArchiveFileSystem
from fsspec.utils import DEFAULT_BLOCK_SIZE
class Lz4FileSystem(BaseCompressedFileFileSystem):
    """Read contents of LZ4 file as a filesystem with one file inside."""
    protocol = 'lz4'
    compression = 'lz4'
    extension = '.lz4'