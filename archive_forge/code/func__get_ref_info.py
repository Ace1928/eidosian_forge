import os
from git.compat import defenc
from git.objects import Object
from git.objects.commit import Commit
from git.util import (
from gitdb.exc import BadObject, BadName
from .log import RefLog
from typing import (
from git.types import Commit_ish, PathLike
@classmethod
def _get_ref_info(cls, repo: 'Repo', ref_path: Union[PathLike, None]) -> Union[Tuple[str, None], Tuple[None, str]]:
    """
        :return: (str(sha), str(target_ref_path)) if available, the sha the file at
            rela_path points to, or None.

            target_ref_path is the reference we point to, or None.
        """
    return cls._get_ref_info_helper(repo, ref_path)