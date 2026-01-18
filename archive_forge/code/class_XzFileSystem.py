import os
from typing import Optional
import fsspec
from fsspec.archive import AbstractArchiveFileSystem
from fsspec.utils import DEFAULT_BLOCK_SIZE
class XzFileSystem(BaseCompressedFileFileSystem):
    """Read contents of .xz (LZMA) file as a filesystem with one file inside."""
    protocol = 'xz'
    compression = 'xz'
    extension = '.xz'