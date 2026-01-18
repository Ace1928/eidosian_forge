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
def _copy_inventory_texts(self):
    inv_keys = self._revision_keys
    inventory_index_map, inventory_indices = self._pack_map_and_index_list('inventory_index')
    inv_nodes = self._index_contents(inventory_indices, inv_keys)
    self.pb.update('Copying inventory texts', 2)
    total_items, readv_group_iter = self._least_readv_node_readv(inv_nodes)
    output_lines = bool(self.revision_ids)
    inv_lines = self._copy_nodes_graph(inventory_index_map, self.new_pack._writer, self.new_pack.inventory_index, readv_group_iter, total_items, output_lines=output_lines)
    if self.revision_ids:
        self._process_inventory_lines(inv_lines)
    else:
        list(inv_lines)
        self._text_filter = None
    if 'pack' in debug.debug_flags:
        trace.mutter('%s: create_pack: inventories copied: %s%s %d items t+%6.3fs', time.ctime(), self._pack_collection._upload_transport.base, self.new_pack.random_name, self.new_pack.inventory_index.key_count(), time.time() - self.new_pack.start_time)