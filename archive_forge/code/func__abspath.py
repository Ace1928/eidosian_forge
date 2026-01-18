import contextlib
import errno
import itertools
import os
from io import BytesIO
from stat import S_IFDIR, S_IFLNK, S_IFREG, S_ISDIR
from .. import transport, urlutils
from ..errors import InProcessTransport, LockError, TransportNotPossible
from ..transport import (AppendBasedFileStream, FileExists, LateReadError,
def _abspath(self, relpath):
    """Generate an internal absolute path."""
    relpath = urlutils.unescape(relpath)
    if relpath[:1] == '/':
        return relpath
    cwd_parts = self._cwd.split('/')
    rel_parts = relpath.split('/')
    r = []
    for i in cwd_parts + rel_parts:
        if i == '..':
            if not r:
                raise ValueError('illegal relpath %r under %r' % (relpath, self._cwd))
            r = r[:-1]
        elif i == '.' or i == '':
            pass
        else:
            r.append(i)
            r = self._symlinks.get('/'.join(r), r)
    return '/' + '/'.join(r)