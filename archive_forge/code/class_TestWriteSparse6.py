import tempfile
from io import BytesIO
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
class TestWriteSparse6:
    """Unit tests for writing graphs in the sparse6 format.

    Most of the test cases were checked against the sparse6 encoder in Sage.

    """

    def test_null_graph(self):
        G = nx.null_graph()
        result = BytesIO()
        nx.write_sparse6(G, result)
        assert result.getvalue() == b'>>sparse6<<:?\n'

    def test_trivial_graph(self):
        G = nx.trivial_graph()
        result = BytesIO()
        nx.write_sparse6(G, result)
        assert result.getvalue() == b'>>sparse6<<:@\n'

    def test_empty_graph(self):
        G = nx.empty_graph(5)
        result = BytesIO()
        nx.write_sparse6(G, result)
        assert result.getvalue() == b'>>sparse6<<:D\n'

    def test_large_empty_graph(self):
        G = nx.empty_graph(68)
        result = BytesIO()
        nx.write_sparse6(G, result)
        assert result.getvalue() == b'>>sparse6<<:~?@C\n'

    def test_very_large_empty_graph(self):
        G = nx.empty_graph(258049)
        result = BytesIO()
        nx.write_sparse6(G, result)
        assert result.getvalue() == b'>>sparse6<<:~~???~?@\n'

    def test_complete_graph(self):
        G = nx.complete_graph(4)
        result = BytesIO()
        nx.write_sparse6(G, result)
        assert result.getvalue() == b'>>sparse6<<:CcKI\n'

    def test_no_header(self):
        G = nx.complete_graph(4)
        result = BytesIO()
        nx.write_sparse6(G, result, header=False)
        assert result.getvalue() == b':CcKI\n'

    def test_padding(self):
        codes = (b':Cdv', b':DaYn', b':EaYnN', b':FaYnL', b':GaYnLz')
        for n, code in enumerate(codes, start=4):
            G = nx.path_graph(n)
            result = BytesIO()
            nx.write_sparse6(G, result, header=False)
            assert result.getvalue() == code + b'\n'

    def test_complete_bipartite(self):
        G = nx.complete_bipartite_graph(6, 9)
        result = BytesIO()
        nx.write_sparse6(G, result)
        expected = b'>>sparse6<<:Nk' + b'?G`cJ' * 9 + b'\n'
        assert result.getvalue() == expected

    def test_read_write_inverse(self):
        for i in list(range(13)) + [31, 47, 62, 63, 64, 72]:
            m = min(2 * i, i * i // 2)
            g = nx.random_graphs.gnm_random_graph(i, m, seed=i)
            gstr = BytesIO()
            nx.write_sparse6(g, gstr, header=False)
            gstr = gstr.getvalue().rstrip()
            g2 = nx.from_sparse6_bytes(gstr)
            assert g2.order() == g.order()
            assert edges_equal(g2.edges(), g.edges())

    def test_no_directed_graphs(self):
        with pytest.raises(nx.NetworkXNotImplemented):
            nx.write_sparse6(nx.DiGraph(), BytesIO())

    def test_write_path(self):
        with tempfile.NamedTemporaryFile() as f:
            fullfilename = f.name
        nx.write_sparse6(nx.null_graph(), fullfilename)
        fh = open(fullfilename, mode='rb')
        assert fh.read() == b'>>sparse6<<:?\n'
        fh.close()
        import os
        os.remove(fullfilename)