import pytest
from networkx.exception import NetworkXError
from networkx.generators.duplication import (
class TestDuplicationDivergenceGraph:
    """Unit tests for the
    :func:`networkx.generators.duplication.duplication_divergence_graph`
    function.

    """

    def test_final_size(self):
        G = duplication_divergence_graph(3, 1)
        assert len(G) == 3
        G = duplication_divergence_graph(3, 1, seed=42)
        assert len(G) == 3

    def test_probability_too_large(self):
        with pytest.raises(NetworkXError):
            duplication_divergence_graph(3, 2)

    def test_probability_too_small(self):
        with pytest.raises(NetworkXError):
            duplication_divergence_graph(3, -1)