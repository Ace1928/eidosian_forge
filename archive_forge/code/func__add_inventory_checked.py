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
def _add_inventory_checked(self, revision_id, inv, parents):
    """Add inv to the repository after checking the inputs.

        This function can be overridden to allow different inventory styles.

        :seealso: add_inventory, for the contract.
        """
    serializer = self._format._serializer
    result = inventory.CHKInventory.from_inventory(self.chk_bytes, inv, maximum_size=serializer.maximum_size, search_key_name=serializer.search_key_name)
    inv_lines = result.to_lines()
    return self._inventory_add_lines(revision_id, parents, inv_lines, check_content=False)