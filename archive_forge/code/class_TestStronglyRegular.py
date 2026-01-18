import networkx as nx
from networkx import is_strongly_regular
class TestStronglyRegular:
    """Unit tests for the :func:`~networkx.is_strongly_regular`
    function.

    """

    def test_cycle_graph(self):
        """Tests that the cycle graph on five vertices is strongly
        regular.

        """
        G = nx.cycle_graph(5)
        assert is_strongly_regular(G)

    def test_petersen_graph(self):
        """Tests that the Petersen graph is strongly regular."""
        G = nx.petersen_graph()
        assert is_strongly_regular(G)

    def test_path_graph(self):
        """Tests that the path graph is not strongly regular."""
        G = nx.path_graph(4)
        assert not is_strongly_regular(G)