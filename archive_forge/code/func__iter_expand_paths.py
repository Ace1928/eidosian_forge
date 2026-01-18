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
@unbare_repo
def _iter_expand_paths(self: 'IndexFile', paths: Sequence[PathLike]) -> Iterator[PathLike]:
    """Expand the directories in list of paths to the corresponding paths accordingly.

        :note:
            git will add items multiple times even if a glob overlapped
            with manually specified paths or if paths where specified multiple
            times - we respect that and do not prune.
        """

    def raise_exc(e: Exception) -> NoReturn:
        raise e
    r = str(self.repo.working_tree_dir)
    rs = r + os.sep
    for path in paths:
        abs_path = str(path)
        if not osp.isabs(abs_path):
            abs_path = osp.join(r, path)
        try:
            st = os.lstat(abs_path)
        except OSError:
            pass
        else:
            if S_ISLNK(st.st_mode):
                yield abs_path.replace(rs, '')
                continue
        if not os.path.exists(abs_path) and ('?' in abs_path or '*' in abs_path or '[' in abs_path):
            resolved_paths = glob.glob(abs_path)
            if abs_path not in resolved_paths:
                for f in self._iter_expand_paths(glob.glob(abs_path)):
                    yield str(f).replace(rs, '')
                continue
        try:
            for root, _dirs, files in os.walk(abs_path, onerror=raise_exc):
                for rela_file in files:
                    yield osp.join(root.replace(rs, ''), rela_file)
        except OSError:
            yield abs_path.replace(rs, '')