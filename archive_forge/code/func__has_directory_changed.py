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
def _has_directory_changed(tree_path: bytes, entry):
    """Check if a directory has changed after getting an error.

    When handling an error trying to create a blob from a path, call this
    function. It will check if the path is a directory. If it's a directory
    and a submodule, check the submodule head to see if it's has changed. If
    not, consider the file as changed as Git tracked a file and not a
    directory.

    Return true if the given path should be considered as changed and False
    otherwise or if the path is not a directory.
    """
    if os.path.exists(os.path.join(tree_path, b'.git')):
        head = read_submodule_head(tree_path)
        if entry.sha != head:
            return True
    else:
        return True
    return False