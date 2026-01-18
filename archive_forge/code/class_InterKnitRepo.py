from typing import Type
from ..lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import transport as _mod_transport
from ..repository import InterRepository, IsInWriteGroupError, Repository
from .repository import RepositoryFormatMetaDir
from .serializer import Serializer
from .vf_repository import (InterSameDataRepository,
class InterKnitRepo(InterSameDataRepository):
    """Optimised code paths between Knit based repositories."""

    @classmethod
    def _get_repo_format_to_test(self):
        return RepositoryFormatKnit1()

    @staticmethod
    def is_compatible(source, target):
        """Be compatible with known Knit formats.

        We don't test for the stores being of specific types because that
        could lead to confusing results, and there is no need to be
        overly general.
        """
        try:
            are_knits = isinstance(source._format, RepositoryFormatKnit) and isinstance(target._format, RepositoryFormatKnit)
        except AttributeError:
            return False
        return are_knits and InterRepository._same_model(source, target)

    def search_missing_revision_ids(self, find_ghosts=True, revision_ids=None, if_present_ids=None, limit=None):
        """See InterRepository.search_missing_revision_ids()."""
        with self.lock_read():
            source_ids_set = self._present_source_revisions_for(revision_ids, if_present_ids)
            target_ids = set(self.target.all_revision_ids())
            possibly_present_revisions = target_ids.intersection(source_ids_set)
            actually_present_revisions = set(self.target._eliminate_revisions_not_present(possibly_present_revisions))
            required_revisions = source_ids_set.difference(actually_present_revisions)
            if revision_ids is not None:
                result_set = required_revisions
            else:
                result_set = set(self.source._eliminate_revisions_not_present(required_revisions))
            if limit is not None:
                topo_ordered = self.source.get_graph().iter_topo_order(result_set)
                result_set = set(itertools.islice(topo_ordered, limit))
            return self.source.revision_ids_to_search_result(result_set)