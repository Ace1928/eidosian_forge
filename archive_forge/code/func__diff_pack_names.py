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
def _diff_pack_names(self):
    """Read the pack names from disk, and compare it to the one in memory.

        :return: (disk_nodes, deleted_nodes, new_nodes)
            disk_nodes    The final set of nodes that should be referenced
            deleted_nodes Nodes which have been removed from when we started
            new_nodes     Nodes that are newly introduced
        """
    disk_nodes = set()
    for index, key, value in self._iter_disk_pack_index():
        disk_nodes.add((key[0].decode('ascii'), value))
    orig_disk_nodes = set(disk_nodes)
    current_nodes = set()
    for name, sizes in self._names.items():
        current_nodes.add((name, b' '.join((b'%d' % size for size in sizes))))
    deleted_nodes = self._packs_at_load - current_nodes
    new_nodes = current_nodes - self._packs_at_load
    disk_nodes.difference_update(deleted_nodes)
    disk_nodes.update(new_nodes)
    return (disk_nodes, deleted_nodes, new_nodes, orig_disk_nodes)