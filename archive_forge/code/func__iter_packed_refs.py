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
def _iter_packed_refs(cls, repo: 'Repo') -> Iterator[Tuple[str, str]]:
    """Return an iterator yielding pairs of sha1/path pairs (as strings)
        for the corresponding refs.

        :note: The packed refs file will be kept open as long as we iterate.
        """
    try:
        with open(cls._get_packed_refs_path(repo), 'rt', encoding='UTF-8') as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('#'):
                    if line.startswith('# pack-refs with:') and 'peeled' not in line:
                        raise TypeError('PackingType of packed-Refs not understood: %r' % line)
                    continue
                if line[0] == '^':
                    continue
                yield cast(Tuple[str, str], tuple(line.split(' ', 1)))
    except OSError:
        return None