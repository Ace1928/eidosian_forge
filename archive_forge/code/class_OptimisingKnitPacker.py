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
class OptimisingKnitPacker(KnitPacker):
    """A packer which spends more time to create better disk layouts."""

    def _revision_node_readv(self, revision_nodes):
        """Return the total revisions and the readv's to issue.

        This sort places revisions in topological order with the ancestors
        after the children.

        :param revision_nodes: The revision index contents for the packs being
            incorporated into the new pack.
        :return: As per _least_readv_node_readv.
        """
        ancestors = {}
        by_key = {}
        for index, key, value, references in revision_nodes:
            ancestors[key] = references[0]
            by_key[key] = (index, value, references)
        order = tsort.topo_sort(ancestors)
        total = len(order)
        requests = []
        for key in reversed(order):
            index, value, references = by_key[key]
            bits = value[1:].split(b' ')
            offset, length = (int(bits[0]), int(bits[1]))
            requests.append((index, [(offset, length)], [(key, value[0:1], references)]))
        return (total, requests)

    def open_pack(self):
        """Open a pack for the pack we are creating."""
        new_pack = super().open_pack()
        new_pack.revision_index.set_optimize(for_size=True)
        new_pack.inventory_index.set_optimize(for_size=True)
        new_pack.text_index.set_optimize(for_size=True)
        new_pack.signature_index.set_optimize(for_size=True)
        return new_pack