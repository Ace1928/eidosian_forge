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
def _fetch_all_revisions(self, revision_ids, pb):
    """Fetch everything for the list of revisions.

        :param revision_ids: The list of revisions to fetch. Must be in
            topological order.
        :param pb: A ProgressTask
        :return: None
        """
    basis_id, basis_tree = self._get_basis(revision_ids[0])
    batch_size = 100
    cache = lru_cache.LRUCache(100)
    cache[basis_id] = basis_tree
    del basis_tree
    hints = []
    a_graph = None
    for offset in range(0, len(revision_ids), batch_size):
        self.target.start_write_group()
        try:
            pb.update(gettext('Transferring revisions'), offset, len(revision_ids))
            batch = revision_ids[offset:offset + batch_size]
            basis_id = self._fetch_batch(batch, basis_id, cache)
        except:
            self.source._safe_to_return_from_cache = False
            self.target.abort_write_group()
            raise
        else:
            hint = self.target.commit_write_group()
            if hint:
                hints.extend(hint)
    if hints and self.target._format.pack_compresses:
        self.target.pack(hint=hints)
    pb.update(gettext('Transferring revisions'), len(revision_ids), len(revision_ids))