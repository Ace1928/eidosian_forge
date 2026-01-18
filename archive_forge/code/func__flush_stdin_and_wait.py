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
def _flush_stdin_and_wait(cls, proc: 'Popen[bytes]', ignore_stdout: bool=False) -> bytes:
    stdin_IO = proc.stdin
    if stdin_IO:
        stdin_IO.flush()
        stdin_IO.close()
    stdout = b''
    if not ignore_stdout and proc.stdout:
        stdout = proc.stdout.read()
    if proc.stdout:
        proc.stdout.close()
        proc.wait()
    return stdout