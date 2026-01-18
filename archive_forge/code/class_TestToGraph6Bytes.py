import tempfile
from io import BytesIO
import pytest
import networkx as nx
import networkx.readwrite.graph6 as g6
from networkx.utils import edges_equal, nodes_equal
class TestToGraph6Bytes:

    def test_null_graph(self):
        G = nx.null_graph()
        assert g6.to_graph6_bytes(G) == b'>>graph6<<?\n'

    def test_trivial_graph(self):
        G = nx.trivial_graph()
        assert g6.to_graph6_bytes(G) == b'>>graph6<<@\n'

    def test_complete_graph(self):
        assert g6.to_graph6_bytes(nx.complete_graph(4)) == b'>>graph6<<C~\n'

    def test_large_complete_graph(self):
        G = nx.complete_graph(67)
        assert g6.to_graph6_bytes(G, header=False) == b'~?@B' + b'~' * 368 + b'w\n'

    def test_no_header(self):
        G = nx.complete_graph(4)
        assert g6.to_graph6_bytes(G, header=False) == b'C~\n'

    def test_complete_bipartite_graph(self):
        G = nx.complete_bipartite_graph(6, 9)
        assert g6.to_graph6_bytes(G, header=False) == b'N??F~z{~Fw^_~?~?^_?\n'

    @pytest.mark.parametrize('G', (nx.MultiGraph(), nx.DiGraph()))
    def test_no_directed_or_multi_graphs(self, G):
        with pytest.raises(nx.NetworkXNotImplemented):
            g6.to_graph6_bytes(G)

    def test_length(self):
        for i in list(range(13)) + [31, 47, 62, 63, 64, 72]:
            G = nx.random_graphs.gnm_random_graph(i, i * i // 4, seed=i)
            gstr = g6.to_graph6_bytes(G, header=False).rstrip()
            assert len(gstr) == ((i - 1) * i // 2 + 5) // 6 + (1 if i < 63 else 4)

    def test_roundtrip(self):
        for i in list(range(13)) + [31, 47, 62, 63, 64, 72]:
            G = nx.random_graphs.gnm_random_graph(i, i * i // 4, seed=i)
            data = g6.to_graph6_bytes(G)
            H = nx.from_graph6_bytes(data.rstrip())
            assert nodes_equal(G.nodes(), H.nodes())
            assert edges_equal(G.edges(), H.edges())

    @pytest.mark.parametrize('edge', ((0, 1), (1, 2), (1, 42)))
    def test_relabeling(self, edge):
        G = nx.Graph([edge])
        assert g6.to_graph6_bytes(G) == b'>>graph6<<A_\n'