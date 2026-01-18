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
def _index_path(self) -> PathLike:
    if self.repo.git_dir:
        return join_path_native(self.repo.git_dir, 'index')
    else:
        raise GitCommandError('No git directory given to join index path')