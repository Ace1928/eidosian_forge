import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
class TestMultilineAdjlist:

    @classmethod
    def setup_class(cls):
        cls.G = nx.Graph(name='test')
        e = [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e'), ('e', 'f'), ('a', 'f')]
        cls.G.add_edges_from(e)
        cls.G.add_node('g')
        cls.DG = nx.DiGraph(cls.G)
        cls.DG.remove_edge('b', 'a')
        cls.DG.remove_edge('b', 'c')
        cls.XG = nx.MultiGraph()
        cls.XG.add_weighted_edges_from([(1, 2, 5), (1, 2, 5), (1, 2, 1), (3, 3, 42)])
        cls.XDG = nx.MultiDiGraph(cls.XG)

    def test_parse_multiline_adjlist(self):
        lines = ['1 2', "b {'weight':3, 'name': 'Frodo'}", 'c {}', 'd 1', "e {'weight':6, 'name': 'Saruman'}"]
        nx.parse_multiline_adjlist(iter(lines))
        with pytest.raises(TypeError):
            nx.parse_multiline_adjlist(iter(lines), nodetype=int)
        nx.parse_multiline_adjlist(iter(lines), edgetype=str)
        with pytest.raises(TypeError):
            nx.parse_multiline_adjlist(iter(lines), nodetype=int)
        lines = ['1 a']
        with pytest.raises(TypeError):
            nx.parse_multiline_adjlist(iter(lines))
        lines = ['a 2']
        with pytest.raises(TypeError):
            nx.parse_multiline_adjlist(iter(lines), nodetype=int)
        lines = ['1 2']
        with pytest.raises(TypeError):
            nx.parse_multiline_adjlist(iter(lines))
        lines = ['1 2', '2 {}']
        with pytest.raises(TypeError):
            nx.parse_multiline_adjlist(iter(lines))

    def test_multiline_adjlist_graph(self):
        G = self.G
        fd, fname = tempfile.mkstemp()
        nx.write_multiline_adjlist(G, fname)
        H = nx.read_multiline_adjlist(fname)
        H2 = nx.read_multiline_adjlist(fname)
        assert H is not H2
        assert nodes_equal(list(H), list(G))
        assert edges_equal(list(H.edges()), list(G.edges()))
        os.close(fd)
        os.unlink(fname)

    def test_multiline_adjlist_digraph(self):
        G = self.DG
        fd, fname = tempfile.mkstemp()
        nx.write_multiline_adjlist(G, fname)
        H = nx.read_multiline_adjlist(fname, create_using=nx.DiGraph())
        H2 = nx.read_multiline_adjlist(fname, create_using=nx.DiGraph())
        assert H is not H2
        assert nodes_equal(list(H), list(G))
        assert edges_equal(list(H.edges()), list(G.edges()))
        os.close(fd)
        os.unlink(fname)

    def test_multiline_adjlist_integers(self):
        fd, fname = tempfile.mkstemp()
        G = nx.convert_node_labels_to_integers(self.G)
        nx.write_multiline_adjlist(G, fname)
        H = nx.read_multiline_adjlist(fname, nodetype=int)
        H2 = nx.read_multiline_adjlist(fname, nodetype=int)
        assert H is not H2
        assert nodes_equal(list(H), list(G))
        assert edges_equal(list(H.edges()), list(G.edges()))
        os.close(fd)
        os.unlink(fname)

    def test_multiline_adjlist_multigraph(self):
        G = self.XG
        fd, fname = tempfile.mkstemp()
        nx.write_multiline_adjlist(G, fname)
        H = nx.read_multiline_adjlist(fname, nodetype=int, create_using=nx.MultiGraph())
        H2 = nx.read_multiline_adjlist(fname, nodetype=int, create_using=nx.MultiGraph())
        assert H is not H2
        assert nodes_equal(list(H), list(G))
        assert edges_equal(list(H.edges()), list(G.edges()))
        os.close(fd)
        os.unlink(fname)

    def test_multiline_adjlist_multidigraph(self):
        G = self.XDG
        fd, fname = tempfile.mkstemp()
        nx.write_multiline_adjlist(G, fname)
        H = nx.read_multiline_adjlist(fname, nodetype=int, create_using=nx.MultiDiGraph())
        H2 = nx.read_multiline_adjlist(fname, nodetype=int, create_using=nx.MultiDiGraph())
        assert H is not H2
        assert nodes_equal(list(H), list(G))
        assert edges_equal(list(H.edges()), list(G.edges()))
        os.close(fd)
        os.unlink(fname)

    def test_multiline_adjlist_delimiter(self):
        fh = io.BytesIO()
        G = nx.path_graph(3)
        nx.write_multiline_adjlist(G, fh, delimiter=':')
        fh.seek(0)
        H = nx.read_multiline_adjlist(fh, nodetype=int, delimiter=':')
        assert nodes_equal(list(H), list(G))
        assert edges_equal(list(H.edges()), list(G.edges()))