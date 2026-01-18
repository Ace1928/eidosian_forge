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
def _present_source_revisions_for(self, revision_ids, if_present_ids=None):
    """Returns set of all revisions in ancestry of revision_ids present in
        the source repo.

        :param revision_ids: if None, all revisions in source are returned.
        :param if_present_ids: like revision_ids, but if any/all of these are
            absent no error is raised.
        """
    if revision_ids is not None or if_present_ids is not None:
        if revision_ids is None:
            revision_ids = set()
        if if_present_ids is None:
            if_present_ids = set()
        revision_ids = set(revision_ids)
        if_present_ids = set(if_present_ids)
        all_wanted_ids = revision_ids.union(if_present_ids)
        graph = self.source.get_graph()
        present_revs = set(graph.get_parent_map(all_wanted_ids))
        missing = revision_ids.difference(present_revs)
        if missing:
            raise errors.NoSuchRevision(self.source, missing.pop())
        found_ids = all_wanted_ids.intersection(present_revs)
        source_ids = [rev_id for rev_id, parents in graph.iter_ancestry(found_ids) if rev_id != _mod_revision.NULL_REVISION and parents is not None]
    else:
        source_ids = self.source.all_revision_ids()
    return set(source_ids)