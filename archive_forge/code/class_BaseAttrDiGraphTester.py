import pytest
import networkx as nx
from networkx.utils import nodes_equal
from .test_graph import BaseAttrGraphTester, BaseGraphTester
from .test_graph import TestEdgeSubgraph as _TestGraphEdgeSubgraph
from .test_graph import TestGraph as _TestGraph
class BaseAttrDiGraphTester(BaseDiGraphTester, BaseAttrGraphTester):

    def test_edges_data(self):
        G = self.K3
        all_edges = [(0, 1, {}), (0, 2, {}), (1, 0, {}), (1, 2, {}), (2, 0, {}), (2, 1, {})]
        assert sorted(G.edges(data=True)) == all_edges
        assert sorted(G.edges(0, data=True)) == all_edges[:2]
        assert sorted(G.edges([0, 1], data=True)) == all_edges[:4]
        with pytest.raises(nx.NetworkXError):
            G.edges(-1, True)

    def test_in_degree_weighted(self):
        G = self.K3.copy()
        G.add_edge(0, 1, weight=0.3, other=1.2)
        assert sorted(G.in_degree(weight='weight')) == [(0, 2), (1, 1.3), (2, 2)]
        assert dict(G.in_degree(weight='weight')) == {0: 2, 1: 1.3, 2: 2}
        assert G.in_degree(1, weight='weight') == 1.3
        assert sorted(G.in_degree(weight='other')) == [(0, 2), (1, 2.2), (2, 2)]
        assert dict(G.in_degree(weight='other')) == {0: 2, 1: 2.2, 2: 2}
        assert G.in_degree(1, weight='other') == 2.2
        assert list(G.in_degree(iter([1]), weight='other')) == [(1, 2.2)]

    def test_out_degree_weighted(self):
        G = self.K3.copy()
        G.add_edge(0, 1, weight=0.3, other=1.2)
        assert sorted(G.out_degree(weight='weight')) == [(0, 1.3), (1, 2), (2, 2)]
        assert dict(G.out_degree(weight='weight')) == {0: 1.3, 1: 2, 2: 2}
        assert G.out_degree(0, weight='weight') == 1.3
        assert sorted(G.out_degree(weight='other')) == [(0, 2.2), (1, 2), (2, 2)]
        assert dict(G.out_degree(weight='other')) == {0: 2.2, 1: 2, 2: 2}
        assert G.out_degree(0, weight='other') == 2.2
        assert list(G.out_degree(iter([0]), weight='other')) == [(0, 2.2)]