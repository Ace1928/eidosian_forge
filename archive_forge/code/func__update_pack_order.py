from .. import errors
from .. import transport as _mod_transport
from ..lazy_import import lazy_import
import time
from breezy import (
from breezy.bzr import (
from breezy.bzr.knit import (
from ..bzr import btree_index
from ..bzr.index import (CombinedGraphIndex, GraphIndex,
from ..bzr.vf_repository import StreamSource
from .knitrepo import KnitRepository
from .pack_repo import (NewPack, PackCommitBuilder, Packer, PackRepository,
def _update_pack_order(self, entries, index_to_pack_map):
    """Determine how we want our packs to be ordered.

        This changes the sort order of the self.packs list so that packs unused
        by 'entries' will be at the end of the list, so that future requests
        can avoid probing them.  Used packs will be at the front of the
        self.packs list, in the order of their first use in 'entries'.

        :param entries: A list of (index, ...) tuples
        :param index_to_pack_map: A mapping from index objects to pack objects.
        """
    packs = []
    seen_indexes = set()
    for entry in entries:
        index = entry[0]
        if index not in seen_indexes:
            packs.append(index_to_pack_map[index])
            seen_indexes.add(index)
    if len(packs) == len(self.packs):
        if 'pack' in debug.debug_flags:
            trace.mutter('Not changing pack list, all packs used.')
        return
    seen_packs = set(packs)
    for pack in self.packs:
        if pack not in seen_packs:
            packs.append(pack)
            seen_packs.add(pack)
    if 'pack' in debug.debug_flags:
        old_names = [p.access_tuple()[1] for p in self.packs]
        new_names = [p.access_tuple()[1] for p in packs]
        trace.mutter('Reordering packs\nfrom: %s\n  to: %s', old_names, new_names)
    self.packs = packs