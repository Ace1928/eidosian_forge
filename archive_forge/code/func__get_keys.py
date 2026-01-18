import itertools
from .. import debug, revision, trace
from ..graph import DictParentsProvider, Graph, invert_parent_map
from ..repository import AbstractSearchResult
def _get_keys(self, graph):
    NULL_REVISION = revision.NULL_REVISION
    keys = [key for key, parents in graph.iter_ancestry(self.heads) if key != NULL_REVISION and parents is not None]
    return keys