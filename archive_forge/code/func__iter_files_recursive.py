import os
from .... import errors
from .... import transport as _mod_transport
from ....bzr import versionedfile
from ....errors import BzrError, UnlistableStore
from ....trace import mutter
def _iter_files_recursive(self):
    """Iterate through the files in the transport."""
    yield from self._transport.iter_files_recursive()