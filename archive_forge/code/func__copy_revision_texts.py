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
def _copy_revision_texts(self):
    if self.revision_ids:
        revision_keys = [(revision_id,) for revision_id in self.revision_ids]
    else:
        revision_keys = None
    revision_index_map, revision_indices = self._pack_map_and_index_list('revision_index')
    revision_nodes = self._index_contents(revision_indices, revision_keys)
    revision_nodes = list(revision_nodes)
    self._update_pack_order(revision_nodes, revision_index_map)
    self.pb.update('Copying revision texts', 1)
    total_items, readv_group_iter = self._revision_node_readv(revision_nodes)
    list(self._copy_nodes_graph(revision_index_map, self.new_pack._writer, self.new_pack.revision_index, readv_group_iter, total_items))
    if 'pack' in debug.debug_flags:
        trace.mutter('%s: create_pack: revisions copied: %s%s %d items t+%6.3fs', time.ctime(), self._pack_collection._upload_transport.base, self.new_pack.random_name, self.new_pack.revision_index.key_count(), time.time() - self.new_pack.start_time)
    self._revision_keys = revision_keys