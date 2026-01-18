from gitdb.exc import (
from git.compat import safe_decode
from git.util import remove_password_if_present
from typing import List, Sequence, Tuple, Union, TYPE_CHECKING
from git.types import PathLike
class RepositoryDirtyError(GitError):
    """Thrown whenever an operation on a repository fails as it has uncommitted changes
    that would be overwritten."""

    def __init__(self, repo: 'Repo', message: str) -> None:
        self.repo = repo
        self.message = message

    def __str__(self) -> str:
        return 'Operation cannot be performed on %r: %s' % (self.repo, self.message)