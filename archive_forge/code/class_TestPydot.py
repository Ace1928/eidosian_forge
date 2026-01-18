import os
import tempfile
from io import StringIO
import pytest
import networkx as nx
from networkx.utils import graphs_equal
@pytest.mark.xfail
class TestPydot:

    def pydot_checks(self, G, prog):
        """
        Validate :mod:`pydot`-based usage of the passed NetworkX graph with the
        passed basename of an external GraphViz command (e.g., `dot`, `neato`).
        """
        G.graph['name'] = 'G'
        G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('A', 'D')])
        G.add_node('E')
        graph_layout = nx.nx_pydot.pydot_layout(G, prog=prog)
        assert isinstance(graph_layout, dict)
        P = nx.nx_pydot.to_pydot(G)
        G2 = G.__class__(nx.nx_pydot.from_pydot(P))
        assert graphs_equal(G, G2)
        fd, fname = tempfile.mkstemp()
        P.write_raw(fname)
        Pin_list = pydot.graph_from_dot_file(path=fname, encoding='utf-8')
        assert len(Pin_list) == 1
        Pin = Pin_list[0]
        n1 = sorted((p.get_name() for p in P.get_node_list()))
        n2 = sorted((p.get_name() for p in Pin.get_node_list()))
        assert n1 == n2
        e1 = sorted(((e.get_source(), e.get_destination()) for e in P.get_edge_list()))
        e2 = sorted(((e.get_source(), e.get_destination()) for e in Pin.get_edge_list()))
        assert e1 == e2
        Hin = nx.nx_pydot.read_dot(fname)
        Hin = G.__class__(Hin)
        assert graphs_equal(G, Hin)
        os.close(fd)
        os.unlink(fname)

    def test_undirected(self):
        self.pydot_checks(nx.Graph(), prog='neato')

    def test_directed(self):
        self.pydot_checks(nx.DiGraph(), prog='dot')

    def test_read_write(self):
        G = nx.MultiGraph()
        G.graph['name'] = 'G'
        G.add_edge('1', '2', key='0')
        fh = StringIO()
        nx.nx_pydot.write_dot(G, fh)
        fh.seek(0)
        H = nx.nx_pydot.read_dot(fh)
        assert graphs_equal(G, H)