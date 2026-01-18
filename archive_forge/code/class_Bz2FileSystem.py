import os
from typing import Optional
import fsspec
from fsspec.archive import AbstractArchiveFileSystem
from fsspec.utils import DEFAULT_BLOCK_SIZE
class Bz2FileSystem(BaseCompressedFileFileSystem):
    """Read contents of BZ2 file as a filesystem with one file inside."""
    protocol = 'bz2'
    compression = 'bz2'
    extension = '.bz2'