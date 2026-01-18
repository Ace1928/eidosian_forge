import os
from .... import errors
from .... import transport as _mod_transport
from ....bzr import versionedfile
from ....errors import BzrError, UnlistableStore
from ....trace import mutter
def _check_fileid(self, fileid):
    if not isinstance(fileid, bytes):
        raise TypeError('Fileids should be bytestrings: {} {!r}'.format(type(fileid), fileid))
    if b'\\' in fileid or b'/' in fileid:
        raise ValueError('invalid store id %r' % fileid)