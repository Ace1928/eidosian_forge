import os
from .... import errors
from .... import transport as _mod_transport
from ....bzr import versionedfile
from ....errors import BzrError, UnlistableStore
from ....trace import mutter
def _id_to_names(self, fileid, suffix):
    """Return the names in the expected order"""
    if suffix is not None:
        fn = self._relpath(fileid, [suffix])
    else:
        fn = self._relpath(fileid)
    fn_gz = fn + '.gz'
    if self._compressed:
        return (fn_gz, fn)
    else:
        return (fn, fn_gz)