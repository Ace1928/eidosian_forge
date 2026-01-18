import os
from .... import errors
from .... import transport as _mod_transport
from ....bzr import versionedfile
from ....errors import BzrError, UnlistableStore
from ....trace import mutter
def _relpath(self, fileid, suffixes=None):
    self._check_fileid(fileid)
    if suffixes:
        for suffix in suffixes:
            if suffix not in self._suffixes:
                raise ValueError('Unregistered suffix %r' % suffix)
            self._check_fileid(suffix.encode('utf-8'))
    else:
        suffixes = []
    path = self._mapper.map((fileid,))
    full_path = '.'.join([path] + suffixes)
    return full_path