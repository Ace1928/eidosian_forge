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
def _create_pack_from_packs(self):
    self.pb.update('Opening pack', 0, 5)
    self.new_pack = self.open_pack()
    new_pack = self.new_pack
    new_pack.set_write_cache_size(1024 * 1024)
    if 'pack' in debug.debug_flags:
        plain_pack_list = ['{}{}'.format(a_pack.pack_transport.base, a_pack.name) for a_pack in self.packs]
        if self.revision_ids is not None:
            rev_count = len(self.revision_ids)
        else:
            rev_count = 'all'
        trace.mutter('%s: create_pack: creating pack from source packs: %s%s %s revisions wanted %s t=0', time.ctime(), self._pack_collection._upload_transport.base, new_pack.random_name, plain_pack_list, rev_count)
    self._copy_revision_texts()
    self._copy_inventory_texts()
    self._copy_text_texts()
    signature_filter = self._revision_keys
    signature_index_map, signature_indices = self._pack_map_and_index_list('signature_index')
    signature_nodes = self._index_contents(signature_indices, signature_filter)
    self.pb.update('Copying signature texts', 4)
    self._copy_nodes(signature_nodes, signature_index_map, new_pack._writer, new_pack.signature_index)
    if 'pack' in debug.debug_flags:
        trace.mutter('%s: create_pack: revision signatures copied: %s%s %d items t+%6.3fs', time.ctime(), self._pack_collection._upload_transport.base, new_pack.random_name, new_pack.signature_index.key_count(), time.time() - new_pack.start_time)
    new_pack._check_references()
    if not self._use_pack(new_pack):
        new_pack.abort()
        return None
    self.pb.update('Finishing pack', 5)
    new_pack.finish()
    self._pack_collection.allocate(new_pack)
    return new_pack