import time
from .. import controldir, debug, errors, osutils
from .. import revision as _mod_revision
from .. import trace, ui
from ..bzr import chk_map, chk_serializer
from ..bzr import index as _mod_index
from ..bzr import inventory, pack, versionedfile
from ..bzr.btree_index import BTreeBuilder, BTreeGraphIndex
from ..bzr.groupcompress import GroupCompressVersionedFiles, _GCGraphIndex
from ..bzr.vf_repository import StreamSource
from .pack_repo import (NewPack, Pack, PackCommitBuilder, Packer,
from .static_tuple import StaticTuple
def _build_vf(self, index_name, parents, delta, for_write=False):
    """Build a VersionedFiles instance on top of this group of packs."""
    index_name = index_name + '_index'
    index_to_pack = {}
    access = _DirectPackAccess(index_to_pack, reload_func=self._reload_func)
    if for_write:
        if self.new_pack is None:
            raise AssertionError('No new pack has been set')
        index = getattr(self.new_pack, index_name)
        index_to_pack[index] = self.new_pack.access_tuple()
        index.set_optimize(for_size=True)
        access.set_writer(self.new_pack._writer, index, self.new_pack.access_tuple())
        add_callback = index.add_nodes
    else:
        indices = []
        for pack in self.packs:
            sub_index = getattr(pack, index_name)
            index_to_pack[sub_index] = pack.access_tuple()
            indices.append(sub_index)
        index = _mod_index.CombinedGraphIndex(indices)
        add_callback = None
    vf = GroupCompressVersionedFiles(_GCGraphIndex(index, add_callback=add_callback, parents=parents, is_locked=self._pack_collection.repo.is_locked), access=access, delta=delta)
    return vf