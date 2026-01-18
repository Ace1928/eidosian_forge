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
def _walk_to_common_revisions(self, revision_ids, if_present_ids=None):
    """Walk out from revision_ids in source to revisions target has.

        :param revision_ids: The start point for the search.
        :return: A set of revision ids.
        """
    target_graph = self.target.get_graph()
    revision_ids = frozenset(revision_ids)
    if if_present_ids:
        all_wanted_revs = revision_ids.union(if_present_ids)
    else:
        all_wanted_revs = revision_ids
    missing_revs = set()
    source_graph = self.source.get_graph()
    searcher = source_graph._make_breadth_first_searcher(all_wanted_revs)
    null_set = frozenset([_mod_revision.NULL_REVISION])
    searcher_exhausted = False
    while True:
        next_revs = set()
        ghosts = set()
        while len(next_revs) < self._walk_to_common_revisions_batch_size:
            try:
                next_revs_part, ghosts_part = searcher.next_with_ghosts()
                next_revs.update(next_revs_part)
                ghosts.update(ghosts_part)
            except StopIteration:
                searcher_exhausted = True
                break
        ghosts_to_check = set(revision_ids.intersection(ghosts))
        revs_to_get = set(next_revs).union(ghosts_to_check)
        if revs_to_get:
            have_revs = set(target_graph.get_parent_map(revs_to_get))
            have_revs = have_revs.union(null_set)
            ghosts_to_check.difference_update(have_revs)
            if ghosts_to_check:
                raise errors.NoSuchRevision(self.source, ghosts_to_check.pop())
            missing_revs.update(next_revs - have_revs)
            stop_revs = searcher.find_seen_ancestors(have_revs)
            searcher.stop_searching_any(stop_revs)
        if searcher_exhausted:
            break
    started_keys, excludes, included_keys = searcher.get_state()
    return vf_search.SearchResult(started_keys, excludes, len(included_keys), included_keys)