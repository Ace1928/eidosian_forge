import itertools
from .. import debug, revision, trace
from ..graph import DictParentsProvider, Graph, invert_parent_map
from ..repository import AbstractSearchResult
class EmptySearchResult(AbstractSearchResult):
    """An empty search result."""

    def is_empty(self):
        return True