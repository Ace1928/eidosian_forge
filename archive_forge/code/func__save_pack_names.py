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
def _save_pack_names(self, clear_obsolete_packs=False, obsolete_packs=None):
    """Save the list of packs.

        This will take out the mutex around the pack names list for the
        duration of the method call. If concurrent updates have been made, a
        three-way merge between the current list and the current in memory list
        is performed.

        :param clear_obsolete_packs: If True, clear out the contents of the
            obsolete_packs directory.
        :param obsolete_packs: Packs that are obsolete once the new pack-names
            file has been written.
        :return: A list of the names saved that were not previously on disk.
        """
    already_obsolete = []
    self.lock_names()
    try:
        builder = self._index_builder_class()
        disk_nodes, deleted_nodes, new_nodes, orig_disk_nodes = self._diff_pack_names()
        for name, value in disk_nodes:
            builder.add_node((name.encode('ascii'),), value)
        self.transport.put_file('pack-names', builder.finish(), mode=self.repo.controldir._get_file_mode())
        self._packs_at_load = disk_nodes
        if clear_obsolete_packs:
            to_preserve = None
            if obsolete_packs:
                to_preserve = {o.name for o in obsolete_packs}
            already_obsolete = self._clear_obsolete_packs(to_preserve)
    finally:
        self._unlock_names()
    self._syncronize_pack_names_from_disk_nodes(disk_nodes)
    if obsolete_packs:
        obsolete_packs = [o for o in obsolete_packs if o.name not in already_obsolete]
        self._obsolete_packs(obsolete_packs)
    return [new_node[0] for new_node in new_nodes]