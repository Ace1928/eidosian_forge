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
def _get_inventory_stream(self, inventory_keys, allow_absent=False):
    """Get a stream of inventory texts.

        When this function returns, self._chk_id_roots and self._chk_p_id_roots
        should be populated.
        """
    self._chk_id_roots = []
    self._chk_p_id_roots = []

    def _filtered_inv_stream():
        id_roots_set = set()
        p_id_roots_set = set()
        source_vf = self.from_repository.inventories
        stream = source_vf.get_record_stream(inventory_keys, 'groupcompress', True)
        for record in stream:
            if record.storage_kind == 'absent':
                if allow_absent:
                    continue
                else:
                    raise errors.NoSuchRevision(self, record.key)
            lines = record.get_bytes_as('lines')
            chk_inv = inventory.CHKInventory.deserialise(None, lines, record.key)
            key = chk_inv.id_to_entry.key()
            if key not in id_roots_set:
                self._chk_id_roots.append(key)
                id_roots_set.add(key)
            p_id_map = chk_inv.parent_id_basename_to_file_id
            if p_id_map is None:
                raise AssertionError('Parent id -> file_id map not set')
            key = p_id_map.key()
            if key not in p_id_roots_set:
                p_id_roots_set.add(key)
                self._chk_p_id_roots.append(key)
            yield record
        id_roots_set.clear()
        p_id_roots_set.clear()
    return ('inventories', _filtered_inv_stream())