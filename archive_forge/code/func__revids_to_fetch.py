import operator
from .. import errors, ui
from ..i18n import gettext
from ..revision import NULL_REVISION
from ..trace import mutter
def _revids_to_fetch(self):
    """Determines the exact revisions needed from self.from_repository to
        install self._last_revision in self.to_repository.

        Returns:
          A SearchResult of some sort.  (Possibly a
          PendingAncestryResult, EmptySearchResult, etc.)
        """
    from . import vf_search
    if self._fetch_spec is not None:
        return self._fetch_spec
    elif self._last_revision == NULL_REVISION:
        return vf_search.EmptySearchResult()
    elif self._last_revision is not None:
        return vf_search.NotInOtherForRevs(self.to_repository, self.from_repository, [self._last_revision], find_ghosts=self.find_ghosts).execute()
    else:
        return vf_search.EverythingNotInOther(self.to_repository, self.from_repository, find_ghosts=self.find_ghosts).execute()