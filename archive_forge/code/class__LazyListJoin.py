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
class _LazyListJoin:
    """An iterable yielding the contents of many lists as one list.

    Each iterator made from this will reflect the current contents of the lists
    at the time the iterator is made.

    This is used by Repository's _make_parents_provider implementation so that
    it is safe to do::

      pp = repo._make_parents_provider()      # uses a list of fallback repos
      pp.add_fallback_repository(other_repo)  # appends to that list
      result = pp.get_parent_map(...)
      # The result will include revs from other_repo
    """

    def __init__(self, *list_parts):
        self.list_parts = list_parts

    def __iter__(self):
        full_list = []
        for list_part in self.list_parts:
            full_list.extend(list_part)
        return iter(full_list)

    def __repr__(self):
        return '{}.{}({})'.format(self.__module__, self.__class__.__name__, self.list_parts)