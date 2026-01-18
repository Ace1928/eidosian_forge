from typing import List, Type, TYPE_CHECKING, Optional, Iterable
from .lazy_import import lazy_import
import time
from breezy import (
from breezy.i18n import gettext
from . import controldir, debug, errors, graph, registry, revision as _mod_revision, ui
from .decorators import only_raises
from .inter import InterObject
from .lock import LogicalLockResult, _RelockDebugMixin
from .revisiontree import RevisionTree
from .trace import (log_exception_quietly, mutter, mutter_callsite, note,
class AbstractSearchResult:
    """The result of a search, describing a set of keys.

    Search results are typically used as the 'fetch_spec' parameter when
    fetching revisions.

    :seealso: AbstractSearch
    """

    def get_recipe(self):
        """Return a recipe that can be used to replay this search.

        The recipe allows reconstruction of the same results at a later date.

        :return: A tuple of `(search_kind_str, *details)`.  The details vary by
            kind of search result.
        """
        raise NotImplementedError(self.get_recipe)

    def get_network_struct(self):
        """Return a tuple that can be transmitted via the HPSS protocol."""
        raise NotImplementedError(self.get_network_struct)

    def get_keys(self):
        """Return the keys found in this search.

        :return: A set of keys.
        """
        raise NotImplementedError(self.get_keys)

    def is_empty(self):
        """Return false if the search lists 1 or more revisions."""
        raise NotImplementedError(self.is_empty)

    def refine(self, seen, referenced):
        """Create a new search by refining this search.

        :param seen: Revisions that have been satisfied.
        :param referenced: Revision references observed while satisfying some
            of this search.
        :return: A search result.
        """
        raise NotImplementedError(self.refine)