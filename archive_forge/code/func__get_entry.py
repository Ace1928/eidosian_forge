import os
import threading
from dulwich.objects import ShaFile, hex_to_sha, sha_to_hex
from .. import bedding
from .. import errors as bzr_errors
from .. import osutils, registry, trace
from ..bzr import btree_index as _mod_btree_index
from ..bzr import index as _mod_index
from ..bzr import versionedfile
from ..transport import FileExists, NoSuchFile, get_transport_from_path
def _get_entry(self, key):
    entries = self._index.iter_entries([key])
    try:
        return next(entries)[2]
    except StopIteration:
        if self._builder is None:
            raise KeyError
        entries = self._builder.iter_entries([key])
        try:
            return next(entries)[2]
        except StopIteration:
            raise KeyError