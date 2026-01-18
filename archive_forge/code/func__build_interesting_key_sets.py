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
def _build_interesting_key_sets(repo, inventory_ids, parent_only_inv_ids):
    result = _InterestingKeyInfo()
    for inv in repo.iter_inventories(inventory_ids, 'unordered'):
        root_key = inv.id_to_entry.key()
        pid_root_key = inv.parent_id_basename_to_file_id.key()
        if inv.revision_id in parent_only_inv_ids:
            result.uninteresting_root_keys.add(root_key)
            result.uninteresting_pid_root_keys.add(pid_root_key)
        else:
            result.interesting_root_keys.add(root_key)
            result.interesting_pid_root_keys.add(pid_root_key)
    return result