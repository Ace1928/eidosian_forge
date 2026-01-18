from .. import errors
from .. import graph as _mod_graph
from .. import tests
from ..revision import NULL_REVISION
from . import TestCaseWithMemoryTransport
def _run_heads_break_deeper(self, graph_dict, search):
    """Run heads on a graph-as-a-dict.

        If the search asks for the parents of b'deeper' the test will fail.
        """

    class stub:
        pass

    def get_parent_map(keys):
        result = {}
        for key in keys:
            if key == b'deeper':
                self.fail('key deeper was accessed')
            result[key] = graph_dict[key]
        return result
    an_obj = stub()
    an_obj.get_parent_map = get_parent_map
    graph = _mod_graph.Graph(an_obj)
    return graph.heads(search)