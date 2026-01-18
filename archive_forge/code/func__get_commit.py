import os
from git.compat import defenc
from git.objects import Object
from git.objects.commit import Commit
from git.util import (
from gitdb.exc import BadObject, BadName
from .log import RefLog
from typing import (
from git.types import Commit_ish, PathLike
def _get_commit(self) -> 'Commit':
    """
        :return:
            Commit object we point to. This works for detached and non-detached
            :class:`SymbolicReference` instances. The symbolic reference will be
            dereferenced recursively.
        """
    obj = self._get_object()
    if obj.type == 'tag':
        obj = obj.object
    if obj.type != Commit.type:
        raise TypeError('Symbolic Reference pointed to object %r, commit was required' % obj)
    return obj