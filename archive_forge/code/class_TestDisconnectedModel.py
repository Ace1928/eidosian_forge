import pyomo.common.unittest as unittest
from pyomo.common.dependencies import networkx as nx, networkx_available
from pyomo.contrib.incidence_analysis.common.dulmage_mendelsohn import (
@unittest.skipUnless(networkx_available, 'networkx is not available')
class TestDisconnectedModel(unittest.TestCase):
    """
    An error is raised if top_nodes are not provided for a disconnected
    graph, which we test here.
    """

    def _construct_graph(self):
        """
        Graph with the following incidence matrix:
        |x x         x|
        |x   x        |
        |      x   x  |
        |        x x  |
        |      x x    |
        |            x|
        |            x|
        """
        N = 7
        top_nodes = list(range(N))
        bot_nodes = list(range(N, 2 * N))
        graph = nx.Graph()
        graph.add_nodes_from(top_nodes, bipartite=0)
        graph.add_nodes_from(bot_nodes, bipartite=1)
        edges = [(0, 0), (0, 1), (0, 6), (1, 0), (1, 2), (2, 3), (2, 5), (3, 4), (3, 5), (4, 3), (4, 4), (5, 6), (6, 6)]
        edges = [(i, j + N) for i, j in edges]
        graph.add_edges_from(edges)
        return (graph, top_nodes)

    def test_graph_dm_partition(self):
        graph, top_nodes = self._construct_graph()
        with self.assertRaises(nx.AmbiguousSolution):
            top_dmp, bot_dmp = dulmage_mendelsohn(graph)
        top_dmp, bot_dmp = dulmage_mendelsohn(graph, top_nodes=top_nodes)
        self.assertFalse(nx.is_connected(graph))
        underconstrained_top = {0, 1}
        underconstrained_bot = {7, 8, 9}
        self.assertEqual(underconstrained_top, set(top_dmp[2]))
        self.assertEqual(underconstrained_bot, set(bot_dmp[0] + bot_dmp[1]))
        overconstrained_top = {5, 6}
        overconstrained_bot = {13}
        self.assertEqual(overconstrained_top, set(top_dmp[0] + top_dmp[1]))
        self.assertEqual(overconstrained_bot, set(bot_dmp[2]))
        wellconstrained_top = {2, 3, 4}
        wellconstrained_bot = {10, 11, 12}
        self.assertEqual(wellconstrained_top, set(top_dmp[3]))
        self.assertEqual(wellconstrained_bot, set(bot_dmp[3]))