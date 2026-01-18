from io import BytesIO
from ..lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import serializer
from breezy.i18n import gettext
from breezy.bzr.testament import Testament
from .. import errors
from ..decorators import only_raises
from ..repository import (CommitBuilder, FetchResult, InterRepository,
from ..trace import mutter, note
from .inventory import ROOT_ID, Inventory, entry_factory
from .inventorytree import InventoryTreeChange
from .repository import MetaDirRepository, RepositoryFormatMetaDir
def _fetch_parent_invs_for_stacking(self, parent_map, cache):
    """Find all parent revisions that are absent, but for which the
        inventory is present, and copy those inventories.

        This is necessary to preserve correctness when the source is stacked
        without fallbacks configured.  (Note that in cases like upgrade the
        source may be not have _fallback_repositories even though it is
        stacked.)
        """
    parent_revs = set(itertools.chain.from_iterable(parent_map.values()))
    present_parents = self.source.get_parent_map(parent_revs)
    absent_parents = parent_revs.difference(present_parents)
    parent_invs_keys_for_stacking = self.source.inventories.get_parent_map(((rev_id,) for rev_id in absent_parents))
    parent_inv_ids = [key[-1] for key in parent_invs_keys_for_stacking]
    for parent_tree in self.source.revision_trees(parent_inv_ids):
        current_revision_id = parent_tree.get_revision_id()
        parents_parents_keys = parent_invs_keys_for_stacking[current_revision_id,]
        parents_parents = [key[-1] for key in parents_parents_keys]
        basis_id = _mod_revision.NULL_REVISION
        basis_tree = self.source.revision_tree(basis_id)
        delta = parent_tree.root_inventory._make_delta(basis_tree.root_inventory)
        self.target.add_inventory_by_delta(basis_id, delta, current_revision_id, parents_parents)
        cache[current_revision_id] = parent_tree