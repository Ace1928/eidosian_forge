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
def _get_delta_for_revision(self, tree, parent_ids, possible_trees):
    """Get the best delta and base for this revision.

        :return: (basis_id, delta)
        """
    deltas = []
    texts_possibly_new_in_tree = set()
    for basis_id, basis_tree in possible_trees:
        delta = tree.root_inventory._make_delta(basis_tree.root_inventory)
        for old_path, new_path, file_id, new_entry in delta:
            if new_path is None:
                continue
            if not new_path:
                continue
            kind = new_entry.kind
            if kind != 'directory' and kind != 'file':
                continue
            texts_possibly_new_in_tree.add((file_id, new_entry.revision))
        deltas.append((len(delta), basis_id, delta))
    deltas.sort()
    return deltas[0][1:]