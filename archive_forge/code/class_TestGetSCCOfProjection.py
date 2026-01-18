import random
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.triangularize import (
from pyomo.common.dependencies import (
import pyomo.common.unittest as unittest
@unittest.skipUnless(networkx_available, 'networkx is not available')
@unittest.skipUnless(scipy_available, 'scipy is not available')
class TestGetSCCOfProjection(unittest.TestCase):

    def test_graph_decomposable_tridiagonal_shuffled(self):
        """
        This is the same graph as in test_decomposable_tridiagonal_shuffled
        below, but now we convert the matrix into a bipartite graph and
        use get_scc_of_projection.

        The matrix decomposes into 2x2 blocks:
        |x x      |
        |x x      |
        |  x x x  |
        |    x x  |
        |      x x|
        """
        N = 11
        row = []
        col = []
        data = []
        row.extend(range(N))
        col.extend(range(N))
        data.extend((1 for _ in range(N)))
        row.extend(range(1, N))
        col.extend(range(N - 1))
        data.extend((1 for _ in range(N - 1)))
        row.extend((i for i in range(N - 1) if not i % 2))
        col.extend((i + 1 for i in range(N - 1) if not i % 2))
        data.extend((1 for i in range(N - 1) if not i % 2))
        row_perm = list(range(N))
        col_perm = list(range(N))
        random.shuffle(row_perm)
        random.shuffle(col_perm)
        row = [row_perm[i] for i in row]
        col = [col_perm[j] for j in col]
        matrix = sps.coo_matrix((data, (row, col)), shape=(N, N))
        graph = nxb.matrix.from_biadjacency_matrix(matrix)
        row_nodes = list(range(N))
        sccs = get_scc_of_projection(graph, row_nodes)
        self.assertEqual(len(sccs), (N + 1) // 2)
        for i in range((N + 1) // 2):
            rows = set((r for r, _ in sccs[i]))
            cols = set((c - N for _, c in sccs[i]))
            pred_rows = {row_perm[2 * i]}
            pred_cols = {col_perm[2 * i]}
            if 2 * i + 1 < N:
                pred_rows.add(row_perm[2 * i + 1])
                pred_cols.add(col_perm[2 * i + 1])
            self.assertEqual(pred_rows, rows)
            self.assertEqual(pred_cols, cols)

    def test_scc_exceptions(self):
        graph = nx.Graph()
        graph.add_nodes_from(range(3))
        graph.add_edges_from([(0, 1), (0, 2), (1, 2)])
        top_nodes = [0]
        msg = 'graph is not bipartite'
        with self.assertRaisesRegex(RuntimeError, msg):
            sccs = get_scc_of_projection(graph, top_nodes=top_nodes)
        graph = nx.Graph()
        graph.add_nodes_from(range(3))
        graph.add_edges_from([(0, 1), (0, 2)])
        top_nodes[0]
        msg = 'bipartite sets of different cardinalities'
        with self.assertRaisesRegex(RuntimeError, msg):
            sccs = get_scc_of_projection(graph, top_nodes=top_nodes)
        graph = nx.Graph()
        graph.add_nodes_from(range(4))
        graph.add_edges_from([(0, 1), (0, 2)])
        top_nodes = [0, 3]
        msg = 'without a perfect matching'
        with self.assertRaisesRegex(RuntimeError, msg):
            sccs = get_scc_of_projection(graph, top_nodes=top_nodes)