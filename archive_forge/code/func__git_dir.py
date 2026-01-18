import os
from git.compat import defenc
from git.objects import Object
from git.objects.commit import Commit
from git.util import (
from gitdb.exc import BadObject, BadName
from .log import RefLog
from typing import (
from git.types import Commit_ish, PathLike
def _git_dir(repo: 'Repo', path: Union[PathLike, None]) -> PathLike:
    """Find the git dir that is appropriate for the path."""
    name = f'{path}'
    if name in ['HEAD', 'ORIG_HEAD', 'FETCH_HEAD', 'index', 'logs']:
        return repo.git_dir
    return repo.common_dir