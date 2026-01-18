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
def _get_trees(self, revision_ids, cache):
    possible_trees = []
    for rev_id in revision_ids:
        if rev_id in cache:
            possible_trees.append((rev_id, cache[rev_id]))
        else:
            try:
                tree = self.source.revision_tree(rev_id)
            except errors.NoSuchRevision:
                pass
            else:
                cache[rev_id] = tree
                possible_trees.append((rev_id, tree))
    return possible_trees