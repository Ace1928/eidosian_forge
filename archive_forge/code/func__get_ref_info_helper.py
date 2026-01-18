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
def _get_ref_info_helper(cls, repo: 'Repo', ref_path: Union[PathLike, None]) -> Union[Tuple[str, None], Tuple[None, str]]:
    """
        :return: (str(sha), str(target_ref_path)) if available, the sha the file at
            rela_path points to, or None.

            target_ref_path is the reference we point to, or None.
        """
    if ref_path:
        cls._check_ref_name_valid(ref_path)
    tokens: Union[None, List[str], Tuple[str, str]] = None
    repodir = _git_dir(repo, ref_path)
    try:
        with open(os.path.join(repodir, str(ref_path)), 'rt', encoding='UTF-8') as fp:
            value = fp.read().rstrip()
        tokens = value.split()
        assert len(tokens) != 0
    except OSError:
        for sha, path in cls._iter_packed_refs(repo):
            if path != ref_path:
                continue
            tokens = (sha, path)
            break
    if tokens is None:
        raise ValueError('Reference at %r does not exist' % ref_path)
    if tokens[0] == 'ref:':
        return (None, tokens[1])
    if repo.re_hexsha_only.match(tokens[0]):
        return (tokens[0], None)
    raise ValueError('Failed to parse reference information from %r' % ref_path)