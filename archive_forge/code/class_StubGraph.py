from ... import graph as _mod_graph
from ... import tests
from ...revision import NULL_REVISION
from ...tests.test_graph import TestGraphBase
from .. import vf_search
class StubGraph:

    def iter_ancestry(self, keys):
        return [(NULL_REVISION, ()), (b'foo', (NULL_REVISION,))]