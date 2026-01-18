from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def _get_cached_parent_map(self, nodes):
    self.calls.append((self.info, 'cached', sorted(nodes)))
    return self._real_parents_provider.get_cached_parent_map(nodes)