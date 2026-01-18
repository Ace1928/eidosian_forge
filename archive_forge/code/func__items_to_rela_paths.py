import contextlib
import datetime
import glob
from io import BytesIO
import os
from stat import S_ISLNK
import subprocess
import tempfile
from git.compat import (
from git.exc import GitCommandError, CheckoutError, GitError, InvalidGitRepositoryError
from git.objects import (
from git.objects.util import Serializable
from git.util import (
from gitdb.base import IStream
from gitdb.db import MemoryDB
import git.diff as git_diff
import os.path as osp
from .fun import (
from .typ import (
from .util import TemporaryFileSwap, post_clear_cache, default_index, git_working_dir
from typing import (
from git.types import Commit_ish, PathLike
def _items_to_rela_paths(self, items: Union[PathLike, Sequence[Union[PathLike, BaseIndexEntry, Blob, Submodule]]]) -> List[PathLike]:
    """Returns a list of repo-relative paths from the given items which
        may be absolute or relative paths, entries or blobs."""
    paths = []
    if isinstance(items, (str, os.PathLike)):
        items = [items]
    for item in items:
        if isinstance(item, (BaseIndexEntry, (Blob, Submodule))):
            paths.append(self._to_relative_path(item.path))
        elif isinstance(item, (str, os.PathLike)):
            paths.append(self._to_relative_path(item))
        else:
            raise TypeError('Invalid item type: %r' % item)
    return paths