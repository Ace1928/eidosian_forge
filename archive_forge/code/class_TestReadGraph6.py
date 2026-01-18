import tempfile
from io import BytesIO
import pytest
import networkx as nx
import networkx.readwrite.graph6 as g6
from networkx.utils import edges_equal, nodes_equal
class TestReadGraph6:

    def test_read_many_graph6(self):
        """Test for reading many graphs from a file into a list."""
        data = b'DF{\nD`{\nDqK\nD~{\n'
        fh = BytesIO(data)
        glist = nx.read_graph6(fh)
        assert len(glist) == 4
        for G in glist:
            assert sorted(G) == list(range(5))