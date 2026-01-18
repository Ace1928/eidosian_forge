import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
class TestWriteGraphML(BaseGraphML):
    writer = staticmethod(nx.write_graphml_lxml)

    @classmethod
    def setup_class(cls):
        BaseGraphML.setup_class()
        _ = pytest.importorskip('lxml.etree')

    def test_write_interface(self):
        try:
            import lxml.etree
            assert nx.write_graphml == nx.write_graphml_lxml
        except ImportError:
            assert nx.write_graphml == nx.write_graphml_xml

    def test_write_read_simple_directed_graphml(self):
        G = self.simple_directed_graph
        G.graph['hi'] = 'there'
        fh = io.BytesIO()
        self.writer(G, fh)
        fh.seek(0)
        H = nx.read_graphml(fh)
        assert sorted(G.nodes()) == sorted(H.nodes())
        assert sorted(G.edges()) == sorted(H.edges())
        assert sorted(G.edges(data=True)) == sorted(H.edges(data=True))
        self.simple_directed_fh.seek(0)

    def test_GraphMLWriter_add_graphs(self):
        gmlw = GraphMLWriter()
        G = self.simple_directed_graph
        H = G.copy()
        gmlw.add_graphs([G, H])

    def test_write_read_simple_no_prettyprint(self):
        G = self.simple_directed_graph
        G.graph['hi'] = 'there'
        G.graph['id'] = '1'
        fh = io.BytesIO()
        self.writer(G, fh, prettyprint=False)
        fh.seek(0)
        H = nx.read_graphml(fh)
        assert sorted(G.nodes()) == sorted(H.nodes())
        assert sorted(G.edges()) == sorted(H.edges())
        assert sorted(G.edges(data=True)) == sorted(H.edges(data=True))
        self.simple_directed_fh.seek(0)

    def test_write_read_attribute_named_key_ids_graphml(self):
        from xml.etree.ElementTree import parse
        G = self.attribute_named_key_ids_graph
        fh = io.BytesIO()
        self.writer(G, fh, named_key_ids=True)
        fh.seek(0)
        H = nx.read_graphml(fh)
        fh.seek(0)
        assert nodes_equal(G.nodes(), H.nodes())
        assert edges_equal(G.edges(), H.edges())
        assert edges_equal(G.edges(data=True), H.edges(data=True))
        self.attribute_named_key_ids_fh.seek(0)
        xml = parse(fh)
        children = list(xml.getroot())
        assert len(children) == 4
        keys = [child.items() for child in children[:3]]
        assert len(keys) == 3
        assert ('id', 'edge_prop') in keys[0]
        assert ('attr.name', 'edge_prop') in keys[0]
        assert ('id', 'prop2') in keys[1]
        assert ('attr.name', 'prop2') in keys[1]
        assert ('id', 'prop1') in keys[2]
        assert ('attr.name', 'prop1') in keys[2]
        default_behavior_fh = io.BytesIO()
        nx.write_graphml(G, default_behavior_fh)
        default_behavior_fh.seek(0)
        H = nx.read_graphml(default_behavior_fh)
        named_key_ids_behavior_fh = io.BytesIO()
        nx.write_graphml(G, named_key_ids_behavior_fh, named_key_ids=True)
        named_key_ids_behavior_fh.seek(0)
        J = nx.read_graphml(named_key_ids_behavior_fh)
        assert all((n1 == n2 for n1, n2 in zip(H.nodes, J.nodes)))
        assert all((e1 == e2 for e1, e2 in zip(H.edges, J.edges)))

    def test_write_read_attribute_numeric_type_graphml(self):
        from xml.etree.ElementTree import parse
        G = self.attribute_numeric_type_graph
        fh = io.BytesIO()
        self.writer(G, fh, infer_numeric_types=True)
        fh.seek(0)
        H = nx.read_graphml(fh)
        fh.seek(0)
        assert nodes_equal(G.nodes(), H.nodes())
        assert edges_equal(G.edges(), H.edges())
        assert edges_equal(G.edges(data=True), H.edges(data=True))
        self.attribute_numeric_type_fh.seek(0)
        xml = parse(fh)
        children = list(xml.getroot())
        assert len(children) == 3
        keys = [child.items() for child in children[:2]]
        assert len(keys) == 2
        assert ('attr.type', 'double') in keys[0]
        assert ('attr.type', 'double') in keys[1]

    def test_more_multigraph_keys(self):
        """Writing keys as edge id attributes means keys become strings.
        The original keys are stored as data, so read them back in
        if `str(key) == edge_id`
        This allows the adjacency to remain the same.
        """
        G = nx.MultiGraph()
        G.add_edges_from([('a', 'b', 2), ('a', 'b', 3)])
        fd, fname = tempfile.mkstemp()
        self.writer(G, fname)
        H = nx.read_graphml(fname)
        assert H.is_multigraph()
        assert edges_equal(G.edges(keys=True), H.edges(keys=True))
        assert G._adj == H._adj
        os.close(fd)
        os.unlink(fname)

    def test_default_attribute(self):
        G = nx.Graph(name='Fred')
        G.add_node(1, label=1, color='green')
        nx.add_path(G, [0, 1, 2, 3])
        G.add_edge(1, 2, weight=3)
        G.graph['node_default'] = {'color': 'yellow'}
        G.graph['edge_default'] = {'weight': 7}
        fh = io.BytesIO()
        self.writer(G, fh)
        fh.seek(0)
        H = nx.read_graphml(fh, node_type=int)
        assert nodes_equal(G.nodes(), H.nodes())
        assert edges_equal(G.edges(), H.edges())
        assert G.graph == H.graph

    def test_mixed_type_attributes(self):
        G = nx.MultiGraph()
        G.add_node('n0', special=False)
        G.add_node('n1', special=0)
        G.add_edge('n0', 'n1', special=False)
        G.add_edge('n0', 'n1', special=0)
        fh = io.BytesIO()
        self.writer(G, fh)
        fh.seek(0)
        H = nx.read_graphml(fh)
        assert not H.nodes['n0']['special']
        assert H.nodes['n1']['special'] == 0
        assert not H.edges['n0', 'n1', 0]['special']
        assert H.edges['n0', 'n1', 1]['special'] == 0

    def test_str_number_mixed_type_attributes(self):
        G = nx.MultiGraph()
        G.add_node('n0', special='hello')
        G.add_node('n1', special=0)
        G.add_edge('n0', 'n1', special='hello')
        G.add_edge('n0', 'n1', special=0)
        fh = io.BytesIO()
        self.writer(G, fh)
        fh.seek(0)
        H = nx.read_graphml(fh)
        assert H.nodes['n0']['special'] == 'hello'
        assert H.nodes['n1']['special'] == 0
        assert H.edges['n0', 'n1', 0]['special'] == 'hello'
        assert H.edges['n0', 'n1', 1]['special'] == 0

    def test_mixed_int_type_number_attributes(self):
        np = pytest.importorskip('numpy')
        G = nx.MultiGraph()
        G.add_node('n0', special=np.int64(0))
        G.add_node('n1', special=1)
        G.add_edge('n0', 'n1', special=np.int64(2))
        G.add_edge('n0', 'n1', special=3)
        fh = io.BytesIO()
        self.writer(G, fh)
        fh.seek(0)
        H = nx.read_graphml(fh)
        assert H.nodes['n0']['special'] == 0
        assert H.nodes['n1']['special'] == 1
        assert H.edges['n0', 'n1', 0]['special'] == 2
        assert H.edges['n0', 'n1', 1]['special'] == 3

    def test_multigraph_to_graph(self):
        G = nx.MultiGraph()
        G.add_edges_from([('a', 'b', 2), ('b', 'c', 3)])
        fd, fname = tempfile.mkstemp()
        self.writer(G, fname)
        H = nx.read_graphml(fname)
        assert not H.is_multigraph()
        H = nx.read_graphml(fname, force_multigraph=True)
        assert H.is_multigraph()
        os.close(fd)
        os.unlink(fname)
        G.add_edge('a', 'b', 'e-id')
        fd, fname = tempfile.mkstemp()
        self.writer(G, fname)
        H = nx.read_graphml(fname)
        assert H.is_multigraph()
        H = nx.read_graphml(fname, force_multigraph=True)
        assert H.is_multigraph()
        os.close(fd)
        os.unlink(fname)

    def test_write_generate_edge_id_from_attribute(self):
        from xml.etree.ElementTree import parse
        G = nx.Graph()
        G.add_edges_from([('a', 'b'), ('b', 'c'), ('a', 'c')])
        edge_attributes = {e: str(e) for e in G.edges}
        nx.set_edge_attributes(G, edge_attributes, 'eid')
        fd, fname = tempfile.mkstemp()
        self.writer(G, fname, edge_id_from_attribute='eid')
        generator = nx.generate_graphml(G, edge_id_from_attribute='eid')
        H = nx.read_graphml(fname)
        assert nodes_equal(G.nodes(), H.nodes())
        assert edges_equal(G.edges(), H.edges())
        nx.set_edge_attributes(G, edge_attributes, 'id')
        assert edges_equal(G.edges(data=True), H.edges(data=True))
        tree = parse(fname)
        children = list(tree.getroot())
        assert len(children) == 2
        edge_ids = [edge.attrib['id'] for edge in tree.getroot().findall('.//{http://graphml.graphdrawing.org/xmlns}edge')]
        assert sorted(edge_ids) == sorted(edge_attributes.values())
        data = ''.join(generator)
        J = nx.parse_graphml(data)
        assert sorted(G.nodes()) == sorted(J.nodes())
        assert sorted(G.edges()) == sorted(J.edges())
        nx.set_edge_attributes(G, edge_attributes, 'id')
        assert edges_equal(G.edges(data=True), J.edges(data=True))
        os.close(fd)
        os.unlink(fname)

    def test_multigraph_write_generate_edge_id_from_attribute(self):
        from xml.etree.ElementTree import parse
        G = nx.MultiGraph()
        G.add_edges_from([('a', 'b'), ('b', 'c'), ('a', 'c'), ('a', 'b')])
        edge_attributes = {e: str(e) for e in G.edges}
        nx.set_edge_attributes(G, edge_attributes, 'eid')
        fd, fname = tempfile.mkstemp()
        self.writer(G, fname, edge_id_from_attribute='eid')
        generator = nx.generate_graphml(G, edge_id_from_attribute='eid')
        H = nx.read_graphml(fname)
        assert H.is_multigraph()
        H = nx.read_graphml(fname, force_multigraph=True)
        assert H.is_multigraph()
        assert nodes_equal(G.nodes(), H.nodes())
        assert edges_equal(G.edges(), H.edges())
        assert sorted((data.get('eid') for u, v, data in H.edges(data=True))) == sorted(edge_attributes.values())
        assert sorted((key for u, v, key in H.edges(keys=True))) == sorted(edge_attributes.values())
        tree = parse(fname)
        children = list(tree.getroot())
        assert len(children) == 2
        edge_ids = [edge.attrib['id'] for edge in tree.getroot().findall('.//{http://graphml.graphdrawing.org/xmlns}edge')]
        assert sorted(edge_ids) == sorted(edge_attributes.values())
        graphml_data = ''.join(generator)
        J = nx.parse_graphml(graphml_data)
        assert J.is_multigraph()
        assert nodes_equal(G.nodes(), J.nodes())
        assert edges_equal(G.edges(), J.edges())
        assert sorted((data.get('eid') for u, v, data in J.edges(data=True))) == sorted(edge_attributes.values())
        assert sorted((key for u, v, key in J.edges(keys=True))) == sorted(edge_attributes.values())
        os.close(fd)
        os.unlink(fname)

    def test_numpy_float64(self):
        np = pytest.importorskip('numpy')
        wt = np.float64(3.4)
        G = nx.Graph([(1, 2, {'weight': wt})])
        fd, fname = tempfile.mkstemp()
        self.writer(G, fname)
        H = nx.read_graphml(fname, node_type=int)
        assert G.edges == H.edges
        wtG = G[1][2]['weight']
        wtH = H[1][2]['weight']
        assert wtG == pytest.approx(wtH, abs=1e-06)
        assert type(wtG) == np.float64
        assert type(wtH) == float
        os.close(fd)
        os.unlink(fname)

    def test_numpy_float32(self):
        np = pytest.importorskip('numpy')
        wt = np.float32(3.4)
        G = nx.Graph([(1, 2, {'weight': wt})])
        fd, fname = tempfile.mkstemp()
        self.writer(G, fname)
        H = nx.read_graphml(fname, node_type=int)
        assert G.edges == H.edges
        wtG = G[1][2]['weight']
        wtH = H[1][2]['weight']
        assert wtG == pytest.approx(wtH, abs=1e-06)
        assert type(wtG) == np.float32
        assert type(wtH) == float
        os.close(fd)
        os.unlink(fname)

    def test_numpy_float64_inference(self):
        np = pytest.importorskip('numpy')
        G = self.attribute_numeric_type_graph
        G.edges['n1', 'n1']['weight'] = np.float64(1.1)
        fd, fname = tempfile.mkstemp()
        self.writer(G, fname, infer_numeric_types=True)
        H = nx.read_graphml(fname)
        assert G._adj == H._adj
        os.close(fd)
        os.unlink(fname)

    def test_unicode_attributes(self):
        G = nx.Graph()
        name1 = chr(2344) + chr(123) + chr(6543)
        name2 = chr(5543) + chr(1543) + chr(324)
        node_type = str
        G.add_edge(name1, 'Radiohead', foo=name2)
        fd, fname = tempfile.mkstemp()
        self.writer(G, fname)
        H = nx.read_graphml(fname, node_type=node_type)
        assert G._adj == H._adj
        os.close(fd)
        os.unlink(fname)

    def test_unicode_escape(self):
        import json
        a = {'a': '{"a": "123"}'}
        sa = json.dumps(a)
        G = nx.Graph()
        G.graph['test'] = sa
        fh = io.BytesIO()
        self.writer(G, fh)
        fh.seek(0)
        H = nx.read_graphml(fh)
        assert G.graph['test'] == H.graph['test']