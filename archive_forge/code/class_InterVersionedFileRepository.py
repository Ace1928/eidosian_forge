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
class InterVersionedFileRepository(InterRepository):
    _walk_to_common_revisions_batch_size = 50
    supports_fetch_spec = True

    def fetch(self, revision_id=None, find_ghosts=False, fetch_spec=None, lossy=False):
        """Fetch the content required to construct revision_id.

        The content is copied from self.source to self.target.

        :param revision_id: if None all content is copied, if NULL_REVISION no
                            content is copied.
        :return: None.
        """
        if lossy:
            raise errors.LossyPushToSameVCS(self.source, self.target)
        if self.target._format.experimental:
            ui.ui_factory.show_user_warning('experimental_format_fetch', from_format=self.source._format, to_format=self.target._format)
        from breezy.bzr.fetch import RepoFetcher
        if self.source._format.network_name() != self.target._format.network_name():
            ui.ui_factory.show_user_warning('cross_format_fetch', from_format=self.source._format, to_format=self.target._format)
        with self.lock_write():
            f = RepoFetcher(to_repository=self.target, from_repository=self.source, last_revision=revision_id, fetch_spec=fetch_spec, find_ghosts=find_ghosts)
            return FetchResult()

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

    def search_missing_revision_ids(self, find_ghosts=True, revision_ids=None, if_present_ids=None, limit=None):
        """Return the revision ids that source has that target does not.

        :param revision_ids: return revision ids included by these
            revision_ids.  NoSuchRevision will be raised if any of these
            revisions are not present.
        :param if_present_ids: like revision_ids, but will not cause
            NoSuchRevision if any of these are absent, instead they will simply
            not be in the result.  This is useful for e.g. finding revisions
            to fetch for tags, which may reference absent revisions.
        :param find_ghosts: If True find missing revisions in deep history
            rather than just finding the surface difference.
        :return: A breezy.graph.SearchResult.
        """
        with self.lock_read():
            if not find_ghosts and (revision_ids is not None or if_present_ids is not None):
                result = self._walk_to_common_revisions(revision_ids, if_present_ids=if_present_ids)
                if limit is None:
                    return result
                result_set = result.get_keys()
            else:
                target_ids = set(self.target.all_revision_ids())
                source_ids = self._present_source_revisions_for(revision_ids, if_present_ids)
                result_set = set(source_ids).difference(target_ids)
            if limit is not None:
                topo_ordered = self.source.get_graph().iter_topo_order(result_set)
                result_set = set(itertools.islice(topo_ordered, limit))
            return self.source.revision_ids_to_search_result(result_set)

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

    @classmethod
    def _get_repo_format_to_test(self):
        return None

    @classmethod
    def is_compatible(cls, source, target):
        return source._format.supports_full_versioned_files and target._format.supports_full_versioned_files