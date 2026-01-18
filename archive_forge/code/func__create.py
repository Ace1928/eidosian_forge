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
def _create(cls: Type[T_References], repo: 'Repo', path: PathLike, resolve: bool, reference: Union['SymbolicReference', str], force: bool, logmsg: Union[str, None]=None) -> T_References:
    """Internal method used to create a new symbolic reference.

        If `resolve` is False, the reference will be taken as is, creating
        a proper symbolic reference. Otherwise it will be resolved to the
        corresponding object and a detached symbolic reference will be created
        instead.
        """
    git_dir = _git_dir(repo, path)
    full_ref_path = cls.to_full_path(path)
    abs_ref_path = os.path.join(git_dir, full_ref_path)
    target = reference
    if resolve:
        target = repo.rev_parse(str(reference))
    if not force and os.path.isfile(abs_ref_path):
        target_data = str(target)
        if isinstance(target, SymbolicReference):
            target_data = str(target.path)
        if not resolve:
            target_data = 'ref: ' + target_data
        with open(abs_ref_path, 'rb') as fd:
            existing_data = fd.read().decode(defenc).strip()
        if existing_data != target_data:
            raise OSError('Reference at %r does already exist, pointing to %r, requested was %r' % (full_ref_path, existing_data, target_data))
    ref = cls(repo, full_ref_path)
    ref.set_reference(target, logmsg)
    return ref