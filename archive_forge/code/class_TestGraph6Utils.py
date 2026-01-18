import tempfile
from io import BytesIO
import pytest
import networkx as nx
import networkx.readwrite.graph6 as g6
from networkx.utils import edges_equal, nodes_equal
class TestGraph6Utils:

    def test_n_data_n_conversion(self):
        for i in [0, 1, 42, 62, 63, 64, 258047, 258048, 7744773, 68719476735]:
            assert g6.data_to_n(g6.n_to_data(i))[0] == i
            assert g6.data_to_n(g6.n_to_data(i))[1] == []
            assert g6.data_to_n(g6.n_to_data(i) + [42, 43])[1] == [42, 43]