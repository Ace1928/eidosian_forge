import tempfile
from io import BytesIO
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
class TestSparseGraph6:

    def test_from_sparse6_bytes(self):
        data = b':Q___eDcdFcDeFcE`GaJ`IaHbKNbLM'
        G = nx.from_sparse6_bytes(data)
        assert nodes_equal(sorted(G.nodes()), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
        assert edges_equal(G.edges(), [(0, 1), (0, 2), (0, 3), (1, 12), (1, 14), (2, 13), (2, 15), (3, 16), (3, 17), (4, 7), (4, 9), (4, 11), (5, 6), (5, 8), (5, 9), (6, 10), (6, 11), (7, 8), (7, 10), (8, 12), (9, 15), (10, 14), (11, 13), (12, 16), (13, 17), (14, 17), (15, 16)])

    def test_from_bytes_multigraph_graph(self):
        graph_data = b':An'
        G = nx.from_sparse6_bytes(graph_data)
        assert type(G) == nx.Graph
        multigraph_data = b':Ab'
        M = nx.from_sparse6_bytes(multigraph_data)
        assert type(M) == nx.MultiGraph

    def test_read_sparse6(self):
        data = b':Q___eDcdFcDeFcE`GaJ`IaHbKNbLM'
        G = nx.from_sparse6_bytes(data)
        fh = BytesIO(data)
        Gin = nx.read_sparse6(fh)
        assert nodes_equal(G.nodes(), Gin.nodes())
        assert edges_equal(G.edges(), Gin.edges())

    def test_read_many_graph6(self):
        data = b':Q___eDcdFcDeFcE`GaJ`IaHbKNbLM\n:Q___dCfDEdcEgcbEGbFIaJ`JaHN`IM'
        fh = BytesIO(data)
        glist = nx.read_sparse6(fh)
        assert len(glist) == 2
        for G in glist:
            assert nodes_equal(G.nodes(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])