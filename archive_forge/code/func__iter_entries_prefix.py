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
def _iter_entries_prefix(self, prefix):
    for entry in self._index.iter_entries_prefix([prefix]):
        yield (entry[1], entry[2])
    if self._builder is not None:
        for entry in self._builder.iter_entries_prefix([prefix]):
            yield (entry[1], entry[2])