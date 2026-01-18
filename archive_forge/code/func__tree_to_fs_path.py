import os
import stat
import struct
import sys
from dataclasses import dataclass
from enum import Enum
from typing import (
from .file import GitFile
from .object_store import iter_tree_contents
from .objects import (
from .pack import ObjectContainer, SHA1Reader, SHA1Writer
def _tree_to_fs_path(root_path: bytes, tree_path: bytes):
    """Convert a git tree path to a file system path.

    Args:
      root_path: Root filesystem path
      tree_path: Git tree path as bytes

    Returns: File system path.
    """
    assert isinstance(tree_path, bytes)
    if os_sep_bytes != b'/':
        sep_corrected_path = tree_path.replace(b'/', os_sep_bytes)
    else:
        sep_corrected_path = tree_path
    return os.path.join(root_path, sep_corrected_path)