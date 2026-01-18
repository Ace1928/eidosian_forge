from git.util import IterableList, join_path
import git.diff as git_diff
from git.util import to_bin_sha
from . import util
from .base import IndexObject, IndexObjUnion
from .blob import Blob
from .submodule.base import Submodule
from .fun import tree_entries_from_data, tree_to_stream
from typing import (
from git.types import PathLike, Literal
def _index_by_name(self, name: str) -> int:
    """:return: index of an item with name, or -1 if not found"""
    for i, t in enumerate(self._cache):
        if t[2] == name:
            return i
    return -1