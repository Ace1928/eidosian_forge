import os
from git.compat import defenc
from git.objects import Object
from git.objects.commit import Commit
from git.util import (
from gitdb.exc import BadObject, BadName
from .log import RefLog
from typing import (
from git.types import Commit_ish, PathLike
@property
def abspath(self) -> PathLike:
    return join_path_native(_git_dir(self.repo, self.path), self.path)