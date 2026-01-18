import pytest
import networkx
import networkx as nx
import networkx.algorithms.regular as reg
import networkx.generators as gen
class TestIsKRegular:

    def test_is_k_regular1(self):
        g = gen.cycle_graph(4)
        assert reg.is_k_regular(g, 2)
        assert not reg.is_k_regular(g, 3)

    def test_is_k_regular2(self):
        g = gen.complete_graph(5)
        assert reg.is_k_regular(g, 4)
        assert not reg.is_k_regular(g, 3)
        assert not reg.is_k_regular(g, 6)

    def test_is_k_regular3(self):
        g = gen.lollipop_graph(5, 5)
        assert not reg.is_k_regular(g, 5)
        assert not reg.is_k_regular(g, 6)