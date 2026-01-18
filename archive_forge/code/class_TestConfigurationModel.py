import pytest
import networkx as nx
class TestConfigurationModel:
    """Unit tests for the :func:`~networkx.configuration_model`
    function.

    """

    def test_empty_degree_sequence(self):
        """Tests that an empty degree sequence yields the null graph."""
        G = nx.configuration_model([])
        assert len(G) == 0

    def test_degree_zero(self):
        """Tests that a degree sequence of all zeros yields the empty
        graph.

        """
        G = nx.configuration_model([0, 0, 0])
        assert len(G) == 3
        assert G.number_of_edges() == 0

    def test_degree_sequence(self):
        """Tests that the degree sequence of the generated graph matches
        the input degree sequence.

        """
        deg_seq = [5, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1]
        G = nx.configuration_model(deg_seq, seed=12345678)
        assert sorted((d for n, d in G.degree()), reverse=True) == [5, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1]
        assert sorted((d for n, d in G.degree(range(len(deg_seq)))), reverse=True) == [5, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1]

    def test_random_seed(self):
        """Tests that each call with the same random seed generates the
        same graph.

        """
        deg_seq = [3] * 12
        G1 = nx.configuration_model(deg_seq, seed=1000)
        G2 = nx.configuration_model(deg_seq, seed=1000)
        assert nx.is_isomorphic(G1, G2)
        G1 = nx.configuration_model(deg_seq, seed=10)
        G2 = nx.configuration_model(deg_seq, seed=10)
        assert nx.is_isomorphic(G1, G2)

    def test_directed_disallowed(self):
        """Tests that attempting to create a configuration model graph
        using a directed graph yields an exception.

        """
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.configuration_model([], create_using=nx.DiGraph())

    def test_odd_degree_sum(self):
        """Tests that a degree sequence whose sum is odd yields an
        exception.

        """
        with pytest.raises(nx.NetworkXError):
            nx.configuration_model([1, 2])