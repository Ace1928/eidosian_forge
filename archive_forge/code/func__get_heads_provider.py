from . import errors
from . import graph as _mod_graph
from . import osutils, ui
def _get_heads_provider(self):
    if self._heads_provider is None:
        self._heads_provider = _mod_graph.KnownGraph(self._parent_map)
    return self._heads_provider