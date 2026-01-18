import re
import sys
from typing import Type
from ..lazy_import import lazy_import
import contextlib
import time
from breezy import (
from breezy.bzr import (
from breezy.bzr.index import (
from .. import errors, lockable_files, lockdir
from .. import transport as _mod_transport
from ..bzr import btree_index, index
from ..decorators import only_raises
from ..lock import LogicalLockResult
from ..repository import RepositoryWriteLockResult, _LazyListJoin
from ..trace import mutter, note, warning
from .repository import MetaDirRepository, RepositoryFormatMetaDir
from .serializer import Serializer
from .vf_repository import (MetaDirVersionedFileRepository,
def _syncronize_pack_names_from_disk_nodes(self, disk_nodes):
    """Given the correct set of pack files, update our saved info.

        :return: (removed, added, modified)
            removed     pack names removed from self._names
            added       pack names added to self._names
            modified    pack names that had changed value
        """
    removed = []
    added = []
    modified = []
    new_names = dict(disk_nodes)
    for pack in self.all_packs():
        if pack.name not in new_names:
            removed.append(pack.name)
            self._remove_pack_from_memory(pack)
    for name, value in disk_nodes:
        sizes = self._parse_index_sizes(value)
        if name in self._names:
            if sizes != self._names[name]:
                self._remove_pack_from_memory(self.get_pack_by_name(name))
                self._names[name] = sizes
                self.get_pack_by_name(name)
                modified.append(name)
        else:
            self._names[name] = sizes
            self.get_pack_by_name(name)
            added.append(name)
    return (removed, added, modified)