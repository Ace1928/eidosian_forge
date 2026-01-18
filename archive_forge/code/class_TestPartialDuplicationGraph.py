import pytest
from networkx.exception import NetworkXError
from networkx.generators.duplication import (
class TestPartialDuplicationGraph:
    """Unit tests for the
    :func:`networkx.generators.duplication.partial_duplication_graph`
    function.

    """

    def test_final_size(self):
        N = 10
        n = 5
        p = 0.5
        q = 0.5
        G = partial_duplication_graph(N, n, p, q)
        assert len(G) == N
        G = partial_duplication_graph(N, n, p, q, seed=42)
        assert len(G) == N

    def test_initial_clique_size(self):
        N = 10
        n = 10
        p = 0.5
        q = 0.5
        G = partial_duplication_graph(N, n, p, q)
        assert len(G) == n

    def test_invalid_initial_size(self):
        with pytest.raises(NetworkXError):
            N = 5
            n = 10
            p = 0.5
            q = 0.5
            G = partial_duplication_graph(N, n, p, q)

    def test_invalid_probabilities(self):
        N = 1
        n = 1
        for p, q in [(0.5, 2), (0.5, -1), (2, 0.5), (-1, 0.5)]:
            args = (N, n, p, q)
            pytest.raises(NetworkXError, partial_duplication_graph, *args)