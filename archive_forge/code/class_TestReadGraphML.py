import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
class TestReadGraphML(BaseGraphML):

    def test_read_simple_directed_graphml(self):
        G = self.simple_directed_graph
        H = nx.read_graphml(self.simple_directed_fh)
        assert sorted(G.nodes()) == sorted(H.nodes())
        assert sorted(G.edges()) == sorted(H.edges())
        assert sorted(G.edges(data=True)) == sorted(H.edges(data=True))
        self.simple_directed_fh.seek(0)
        PG = nx.parse_graphml(self.simple_directed_data)
        assert sorted(G.nodes()) == sorted(PG.nodes())
        assert sorted(G.edges()) == sorted(PG.edges())
        assert sorted(G.edges(data=True)) == sorted(PG.edges(data=True))

    def test_read_simple_undirected_graphml(self):
        G = self.simple_undirected_graph
        H = nx.read_graphml(self.simple_undirected_fh)
        assert nodes_equal(G.nodes(), H.nodes())
        assert edges_equal(G.edges(), H.edges())
        self.simple_undirected_fh.seek(0)
        PG = nx.parse_graphml(self.simple_undirected_data)
        assert nodes_equal(G.nodes(), PG.nodes())
        assert edges_equal(G.edges(), PG.edges())

    def test_read_undirected_multigraph_graphml(self):
        G = self.undirected_multigraph
        H = nx.read_graphml(self.undirected_multigraph_fh)
        assert nodes_equal(G.nodes(), H.nodes())
        assert edges_equal(G.edges(), H.edges())
        self.undirected_multigraph_fh.seek(0)
        PG = nx.parse_graphml(self.undirected_multigraph_data)
        assert nodes_equal(G.nodes(), PG.nodes())
        assert edges_equal(G.edges(), PG.edges())

    def test_read_undirected_multigraph_no_multiedge_graphml(self):
        G = self.undirected_multigraph_no_multiedge
        H = nx.read_graphml(self.undirected_multigraph_no_multiedge_fh)
        assert nodes_equal(G.nodes(), H.nodes())
        assert edges_equal(G.edges(), H.edges())
        self.undirected_multigraph_no_multiedge_fh.seek(0)
        PG = nx.parse_graphml(self.undirected_multigraph_no_multiedge_data)
        assert nodes_equal(G.nodes(), PG.nodes())
        assert edges_equal(G.edges(), PG.edges())

    def test_read_undirected_multigraph_only_ids_for_multiedges_graphml(self):
        G = self.multigraph_only_ids_for_multiedges
        H = nx.read_graphml(self.multigraph_only_ids_for_multiedges_fh)
        assert nodes_equal(G.nodes(), H.nodes())
        assert edges_equal(G.edges(), H.edges())
        self.multigraph_only_ids_for_multiedges_fh.seek(0)
        PG = nx.parse_graphml(self.multigraph_only_ids_for_multiedges_data)
        assert nodes_equal(G.nodes(), PG.nodes())
        assert edges_equal(G.edges(), PG.edges())

    def test_read_attribute_graphml(self):
        G = self.attribute_graph
        H = nx.read_graphml(self.attribute_fh)
        assert nodes_equal(G.nodes(True), sorted(H.nodes(data=True)))
        ge = sorted(G.edges(data=True))
        he = sorted(H.edges(data=True))
        for a, b in zip(ge, he):
            assert a == b
        self.attribute_fh.seek(0)
        PG = nx.parse_graphml(self.attribute_data)
        assert sorted(G.nodes(True)) == sorted(PG.nodes(data=True))
        ge = sorted(G.edges(data=True))
        he = sorted(PG.edges(data=True))
        for a, b in zip(ge, he):
            assert a == b

    def test_node_default_attribute_graphml(self):
        G = self.node_attribute_default_graph
        H = nx.read_graphml(self.node_attribute_default_fh)
        assert G.graph['node_default'] == H.graph['node_default']

    def test_directed_edge_in_undirected(self):
        s = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <graph id="G">\n    <node id="n0"/>\n    <node id="n1"/>\n    <node id="n2"/>\n    <edge source="n0" target="n1"/>\n    <edge source="n1" target="n2" directed=\'true\'/>\n  </graph>\n</graphml>'
        fh = io.BytesIO(s.encode('UTF-8'))
        pytest.raises(nx.NetworkXError, nx.read_graphml, fh)
        pytest.raises(nx.NetworkXError, nx.parse_graphml, s)

    def test_undirected_edge_in_directed(self):
        s = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <graph id="G" edgedefault=\'directed\'>\n    <node id="n0"/>\n    <node id="n1"/>\n    <node id="n2"/>\n    <edge source="n0" target="n1"/>\n    <edge source="n1" target="n2" directed=\'false\'/>\n  </graph>\n</graphml>'
        fh = io.BytesIO(s.encode('UTF-8'))
        pytest.raises(nx.NetworkXError, nx.read_graphml, fh)
        pytest.raises(nx.NetworkXError, nx.parse_graphml, s)

    def test_key_raise(self):
        s = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <key id="d0" for="node" attr.name="color" attr.type="string">\n    <default>yellow</default>\n  </key>\n  <key id="d1" for="edge" attr.name="weight" attr.type="double"/>\n  <graph id="G" edgedefault="directed">\n    <node id="n0">\n      <data key="d0">green</data>\n    </node>\n    <node id="n1"/>\n    <node id="n2">\n      <data key="d0">blue</data>\n    </node>\n    <edge id="e0" source="n0" target="n2">\n      <data key="d2">1.0</data>\n    </edge>\n  </graph>\n</graphml>\n'
        fh = io.BytesIO(s.encode('UTF-8'))
        pytest.raises(nx.NetworkXError, nx.read_graphml, fh)
        pytest.raises(nx.NetworkXError, nx.parse_graphml, s)

    def test_hyperedge_raise(self):
        s = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <key id="d0" for="node" attr.name="color" attr.type="string">\n    <default>yellow</default>\n  </key>\n  <key id="d1" for="edge" attr.name="weight" attr.type="double"/>\n  <graph id="G" edgedefault="directed">\n    <node id="n0">\n      <data key="d0">green</data>\n    </node>\n    <node id="n1"/>\n    <node id="n2">\n      <data key="d0">blue</data>\n    </node>\n    <hyperedge id="e0" source="n0" target="n2">\n       <endpoint node="n0"/>\n       <endpoint node="n1"/>\n       <endpoint node="n2"/>\n    </hyperedge>\n  </graph>\n</graphml>\n'
        fh = io.BytesIO(s.encode('UTF-8'))
        pytest.raises(nx.NetworkXError, nx.read_graphml, fh)
        pytest.raises(nx.NetworkXError, nx.parse_graphml, s)

    def test_multigraph_keys(self):
        s = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <graph id="G" edgedefault="directed">\n    <node id="n0"/>\n    <node id="n1"/>\n    <edge id="e0" source="n0" target="n1"/>\n    <edge id="e1" source="n0" target="n1"/>\n  </graph>\n</graphml>\n'
        fh = io.BytesIO(s.encode('UTF-8'))
        G = nx.read_graphml(fh)
        expected = [('n0', 'n1', 'e0'), ('n0', 'n1', 'e1')]
        assert sorted(G.edges(keys=True)) == expected
        fh.seek(0)
        H = nx.parse_graphml(s)
        assert sorted(H.edges(keys=True)) == expected

    def test_preserve_multi_edge_data(self):
        """
        Test that data and keys of edges are preserved on consequent
        write and reads
        """
        G = nx.MultiGraph()
        G.add_node(1)
        G.add_node(2)
        G.add_edges_from([(1, 2), (1, 2, {'key': 'data_key1'}), (1, 2, {'id': 'data_id2'}), (1, 2, {'key': 'data_key3', 'id': 'data_id3'}), (1, 2, 103, {'key': 'data_key4'}), (1, 2, 104, {'id': 'data_id5'}), (1, 2, 105, {'key': 'data_key6', 'id': 'data_id7'})])
        fh = io.BytesIO()
        nx.write_graphml(G, fh)
        fh.seek(0)
        H = nx.read_graphml(fh, node_type=int)
        assert edges_equal(G.edges(data=True, keys=True), H.edges(data=True, keys=True))
        assert G._adj == H._adj
        Gadj = {str(node): {str(nbr): {str(ekey): dd for ekey, dd in key_dict.items()} for nbr, key_dict in nbr_dict.items()} for node, nbr_dict in G._adj.items()}
        fh.seek(0)
        HH = nx.read_graphml(fh, node_type=str, edge_key_type=str)
        assert Gadj == HH._adj
        fh.seek(0)
        string_fh = fh.read()
        HH = nx.parse_graphml(string_fh, node_type=str, edge_key_type=str)
        assert Gadj == HH._adj

    def test_yfiles_extension(self):
        data = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xmlns:y="http://www.yworks.com/xml/graphml"\n         xmlns:yed="http://www.yworks.com/xml/yed/3"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <!--Created by yFiles for Java 2.7-->\n  <key for="graphml" id="d0" yfiles.type="resources"/>\n  <key attr.name="url" attr.type="string" for="node" id="d1"/>\n  <key attr.name="description" attr.type="string" for="node" id="d2"/>\n  <key for="node" id="d3" yfiles.type="nodegraphics"/>\n  <key attr.name="Description" attr.type="string" for="graph" id="d4">\n    <default/>\n  </key>\n  <key attr.name="url" attr.type="string" for="edge" id="d5"/>\n  <key attr.name="description" attr.type="string" for="edge" id="d6"/>\n  <key for="edge" id="d7" yfiles.type="edgegraphics"/>\n  <graph edgedefault="directed" id="G">\n    <node id="n0">\n      <data key="d3">\n        <y:ShapeNode>\n          <y:Geometry height="30.0" width="30.0" x="125.0" y="100.0"/>\n          <y:Fill color="#FFCC00" transparent="false"/>\n          <y:BorderStyle color="#000000" type="line" width="1.0"/>\n          <y:NodeLabel alignment="center" autoSizePolicy="content"\n           borderDistance="0.0" fontFamily="Dialog" fontSize="13"\n           fontStyle="plain" hasBackgroundColor="false" hasLineColor="false"\n           height="19.1328125" modelName="internal" modelPosition="c"\n           textColor="#000000" visible="true" width="12.27099609375"\n           x="8.864501953125" y="5.43359375">1</y:NodeLabel>\n          <y:Shape type="rectangle"/>\n        </y:ShapeNode>\n      </data>\n    </node>\n    <node id="n1">\n      <data key="d3">\n        <y:ShapeNode>\n          <y:Geometry height="30.0" width="30.0" x="183.0" y="205.0"/>\n          <y:Fill color="#FFCC00" transparent="false"/>\n          <y:BorderStyle color="#000000" type="line" width="1.0"/>\n          <y:NodeLabel alignment="center" autoSizePolicy="content"\n          borderDistance="0.0" fontFamily="Dialog" fontSize="13"\n          fontStyle="plain" hasBackgroundColor="false" hasLineColor="false"\n          height="19.1328125" modelName="internal" modelPosition="c"\n          textColor="#000000" visible="true" width="12.27099609375"\n          x="8.864501953125" y="5.43359375">2</y:NodeLabel>\n          <y:Shape type="rectangle"/>\n        </y:ShapeNode>\n      </data>\n    </node>\n    <node id="n2">\n      <data key="d6" xml:space="preserve"><![CDATA[description\nline1\nline2]]></data>\n      <data key="d3">\n        <y:GenericNode configuration="com.yworks.flowchart.terminator">\n          <y:Geometry height="40.0" width="80.0" x="950.0" y="286.0"/>\n          <y:Fill color="#E8EEF7" color2="#B7C9E3" transparent="false"/>\n          <y:BorderStyle color="#000000" type="line" width="1.0"/>\n          <y:NodeLabel alignment="center" autoSizePolicy="content"\n          fontFamily="Dialog" fontSize="12" fontStyle="plain"\n          hasBackgroundColor="false" hasLineColor="false" height="17.96875"\n          horizontalTextPosition="center" iconTextGap="4" modelName="custom"\n          textColor="#000000" verticalTextPosition="bottom" visible="true"\n          width="67.984375" x="6.0078125" xml:space="preserve"\n          y="11.015625">3<y:LabelModel>\n          <y:SmartNodeLabelModel distance="4.0"/></y:LabelModel>\n          <y:ModelParameter><y:SmartNodeLabelModelParameter labelRatioX="0.0"\n          labelRatioY="0.0" nodeRatioX="0.0" nodeRatioY="0.0" offsetX="0.0"\n          offsetY="0.0" upX="0.0" upY="-1.0"/></y:ModelParameter></y:NodeLabel>\n        </y:GenericNode>\n      </data>\n    </node>\n    <edge id="e0" source="n0" target="n1">\n      <data key="d7">\n        <y:PolyLineEdge>\n          <y:Path sx="0.0" sy="0.0" tx="0.0" ty="0.0"/>\n          <y:LineStyle color="#000000" type="line" width="1.0"/>\n          <y:Arrows source="none" target="standard"/>\n          <y:BendStyle smoothed="false"/>\n        </y:PolyLineEdge>\n      </data>\n    </edge>\n  </graph>\n  <data key="d0">\n    <y:Resources/>\n  </data>\n</graphml>\n'
        fh = io.BytesIO(data.encode('UTF-8'))
        G = nx.read_graphml(fh, force_multigraph=True)
        assert list(G.edges()) == [('n0', 'n1')]
        assert G.has_edge('n0', 'n1', key='e0')
        assert G.nodes['n0']['label'] == '1'
        assert G.nodes['n1']['label'] == '2'
        assert G.nodes['n2']['label'] == '3'
        assert G.nodes['n0']['shape_type'] == 'rectangle'
        assert G.nodes['n1']['shape_type'] == 'rectangle'
        assert G.nodes['n2']['shape_type'] == 'com.yworks.flowchart.terminator'
        assert G.nodes['n2']['description'] == 'description\nline1\nline2'
        fh.seek(0)
        G = nx.read_graphml(fh)
        assert list(G.edges()) == [('n0', 'n1')]
        assert G['n0']['n1']['id'] == 'e0'
        assert G.nodes['n0']['label'] == '1'
        assert G.nodes['n1']['label'] == '2'
        assert G.nodes['n2']['label'] == '3'
        assert G.nodes['n0']['shape_type'] == 'rectangle'
        assert G.nodes['n1']['shape_type'] == 'rectangle'
        assert G.nodes['n2']['shape_type'] == 'com.yworks.flowchart.terminator'
        assert G.nodes['n2']['description'] == 'description\nline1\nline2'
        H = nx.parse_graphml(data, force_multigraph=True)
        assert list(H.edges()) == [('n0', 'n1')]
        assert H.has_edge('n0', 'n1', key='e0')
        assert H.nodes['n0']['label'] == '1'
        assert H.nodes['n1']['label'] == '2'
        assert H.nodes['n2']['label'] == '3'
        H = nx.parse_graphml(data)
        assert list(H.edges()) == [('n0', 'n1')]
        assert H['n0']['n1']['id'] == 'e0'
        assert H.nodes['n0']['label'] == '1'
        assert H.nodes['n1']['label'] == '2'
        assert H.nodes['n2']['label'] == '3'

    def test_bool(self):
        s = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <key id="d0" for="node" attr.name="test" attr.type="boolean">\n    <default>false</default>\n  </key>\n  <graph id="G" edgedefault="directed">\n    <node id="n0">\n      <data key="d0">true</data>\n    </node>\n    <node id="n1"/>\n    <node id="n2">\n      <data key="d0">false</data>\n    </node>\n    <node id="n3">\n      <data key="d0">FaLsE</data>\n    </node>\n    <node id="n4">\n      <data key="d0">True</data>\n    </node>\n    <node id="n5">\n      <data key="d0">0</data>\n    </node>\n    <node id="n6">\n      <data key="d0">1</data>\n    </node>\n  </graph>\n</graphml>\n'
        fh = io.BytesIO(s.encode('UTF-8'))
        G = nx.read_graphml(fh)
        H = nx.parse_graphml(s)
        for graph in [G, H]:
            assert graph.nodes['n0']['test']
            assert not graph.nodes['n2']['test']
            assert not graph.nodes['n3']['test']
            assert graph.nodes['n4']['test']
            assert not graph.nodes['n5']['test']
            assert graph.nodes['n6']['test']

    def test_graphml_header_line(self):
        good = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <key id="d0" for="node" attr.name="test" attr.type="boolean">\n    <default>false</default>\n  </key>\n  <graph id="G">\n    <node id="n0">\n      <data key="d0">true</data>\n    </node>\n  </graph>\n</graphml>\n'
        bad = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<graphml>\n  <key id="d0" for="node" attr.name="test" attr.type="boolean">\n    <default>false</default>\n  </key>\n  <graph id="G">\n    <node id="n0">\n      <data key="d0">true</data>\n    </node>\n  </graph>\n</graphml>\n'
        ugly = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<graphml xmlns="https://ghghgh">\n  <key id="d0" for="node" attr.name="test" attr.type="boolean">\n    <default>false</default>\n  </key>\n  <graph id="G">\n    <node id="n0">\n      <data key="d0">true</data>\n    </node>\n  </graph>\n</graphml>\n'
        for s in (good, bad):
            fh = io.BytesIO(s.encode('UTF-8'))
            G = nx.read_graphml(fh)
            H = nx.parse_graphml(s)
            for graph in [G, H]:
                assert graph.nodes['n0']['test']
        fh = io.BytesIO(ugly.encode('UTF-8'))
        pytest.raises(nx.NetworkXError, nx.read_graphml, fh)
        pytest.raises(nx.NetworkXError, nx.parse_graphml, ugly)

    def test_read_attributes_with_groups(self):
        data = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:java="http://www.yworks.com/xml/yfiles-common/1.0/java" xmlns:sys="http://www.yworks.com/xml/yfiles-common/markup/primitives/2.0" xmlns:x="http://www.yworks.com/xml/yfiles-common/markup/2.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:y="http://www.yworks.com/xml/graphml" xmlns:yed="http://www.yworks.com/xml/yed/3" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://www.yworks.com/xml/schema/graphml/1.1/ygraphml.xsd">\n  <!--Created by yEd 3.17-->\n  <key attr.name="Description" attr.type="string" for="graph" id="d0"/>\n  <key for="port" id="d1" yfiles.type="portgraphics"/>\n  <key for="port" id="d2" yfiles.type="portgeometry"/>\n  <key for="port" id="d3" yfiles.type="portuserdata"/>\n  <key attr.name="CustomProperty" attr.type="string" for="node" id="d4">\n    <default/>\n  </key>\n  <key attr.name="url" attr.type="string" for="node" id="d5"/>\n  <key attr.name="description" attr.type="string" for="node" id="d6"/>\n  <key for="node" id="d7" yfiles.type="nodegraphics"/>\n  <key for="graphml" id="d8" yfiles.type="resources"/>\n  <key attr.name="url" attr.type="string" for="edge" id="d9"/>\n  <key attr.name="description" attr.type="string" for="edge" id="d10"/>\n  <key for="edge" id="d11" yfiles.type="edgegraphics"/>\n  <graph edgedefault="directed" id="G">\n    <data key="d0"/>\n    <node id="n0">\n      <data key="d4"><![CDATA[CustomPropertyValue]]></data>\n      <data key="d6"/>\n      <data key="d7">\n        <y:ShapeNode>\n          <y:Geometry height="30.0" width="30.0" x="125.0" y="-255.4611111111111"/>\n          <y:Fill color="#FFCC00" transparent="false"/>\n          <y:BorderStyle color="#000000" raised="false" type="line" width="1.0"/>\n          <y:NodeLabel alignment="center" autoSizePolicy="content" fontFamily="Dialog" fontSize="12" fontStyle="plain" hasBackgroundColor="false" hasLineColor="false" height="17.96875" horizontalTextPosition="center" iconTextGap="4" modelName="custom" textColor="#000000" verticalTextPosition="bottom" visible="true" width="11.634765625" x="9.1826171875" y="6.015625">2<y:LabelModel>\n              <y:SmartNodeLabelModel distance="4.0"/>\n            </y:LabelModel>\n            <y:ModelParameter>\n              <y:SmartNodeLabelModelParameter labelRatioX="0.0" labelRatioY="0.0" nodeRatioX="0.0" nodeRatioY="0.0" offsetX="0.0" offsetY="0.0" upX="0.0" upY="-1.0"/>\n            </y:ModelParameter>\n          </y:NodeLabel>\n          <y:Shape type="rectangle"/>\n        </y:ShapeNode>\n      </data>\n    </node>\n    <node id="n1" yfiles.foldertype="group">\n      <data key="d4"><![CDATA[CustomPropertyValue]]></data>\n      <data key="d5"/>\n      <data key="d6"/>\n      <data key="d7">\n        <y:ProxyAutoBoundsNode>\n          <y:Realizers active="0">\n            <y:GroupNode>\n              <y:Geometry height="250.38333333333333" width="140.0" x="-30.0" y="-330.3833333333333"/>\n              <y:Fill color="#F5F5F5" transparent="false"/>\n              <y:BorderStyle color="#000000" type="dashed" width="1.0"/>\n              <y:NodeLabel alignment="right" autoSizePolicy="node_width" backgroundColor="#EBEBEB" borderDistance="0.0" fontFamily="Dialog" fontSize="15" fontStyle="plain" hasLineColor="false" height="21.4609375" horizontalTextPosition="center" iconTextGap="4" modelName="internal" modelPosition="t" textColor="#000000" verticalTextPosition="bottom" visible="true" width="140.0" x="0.0" y="0.0">Group 3</y:NodeLabel>\n              <y:Shape type="roundrectangle"/>\n              <y:State closed="false" closedHeight="50.0" closedWidth="50.0" innerGraphDisplayEnabled="false"/>\n              <y:Insets bottom="15" bottomF="15.0" left="15" leftF="15.0" right="15" rightF="15.0" top="15" topF="15.0"/>\n              <y:BorderInsets bottom="1" bottomF="1.0" left="0" leftF="0.0" right="0" rightF="0.0" top="1" topF="1.0001736111111086"/>\n            </y:GroupNode>\n            <y:GroupNode>\n              <y:Geometry height="50.0" width="50.0" x="0.0" y="60.0"/>\n              <y:Fill color="#F5F5F5" transparent="false"/>\n              <y:BorderStyle color="#000000" type="dashed" width="1.0"/>\n              <y:NodeLabel alignment="right" autoSizePolicy="node_width" backgroundColor="#EBEBEB" borderDistance="0.0" fontFamily="Dialog" fontSize="15" fontStyle="plain" hasLineColor="false" height="21.4609375" horizontalTextPosition="center" iconTextGap="4" modelName="internal" modelPosition="t" textColor="#000000" verticalTextPosition="bottom" visible="true" width="65.201171875" x="-7.6005859375" y="0.0">Folder 3</y:NodeLabel>\n              <y:Shape type="roundrectangle"/>\n              <y:State closed="true" closedHeight="50.0" closedWidth="50.0" innerGraphDisplayEnabled="false"/>\n              <y:Insets bottom="5" bottomF="5.0" left="5" leftF="5.0" right="5" rightF="5.0" top="5" topF="5.0"/>\n              <y:BorderInsets bottom="0" bottomF="0.0" left="0" leftF="0.0" right="0" rightF="0.0" top="0" topF="0.0"/>\n            </y:GroupNode>\n          </y:Realizers>\n        </y:ProxyAutoBoundsNode>\n      </data>\n      <graph edgedefault="directed" id="n1:">\n        <node id="n1::n0" yfiles.foldertype="group">\n          <data key="d4"><![CDATA[CustomPropertyValue]]></data>\n          <data key="d5"/>\n          <data key="d6"/>\n          <data key="d7">\n            <y:ProxyAutoBoundsNode>\n              <y:Realizers active="0">\n                <y:GroupNode>\n                  <y:Geometry height="83.46111111111111" width="110.0" x="-15.0" y="-292.9222222222222"/>\n                  <y:Fill color="#F5F5F5" transparent="false"/>\n                  <y:BorderStyle color="#000000" type="dashed" width="1.0"/>\n                  <y:NodeLabel alignment="right" autoSizePolicy="node_width" backgroundColor="#EBEBEB" borderDistance="0.0" fontFamily="Dialog" fontSize="15" fontStyle="plain" hasLineColor="false" height="21.4609375" horizontalTextPosition="center" iconTextGap="4" modelName="internal" modelPosition="t" textColor="#000000" verticalTextPosition="bottom" visible="true" width="110.0" x="0.0" y="0.0">Group 1</y:NodeLabel>\n                  <y:Shape type="roundrectangle"/>\n                  <y:State closed="false" closedHeight="50.0" closedWidth="50.0" innerGraphDisplayEnabled="false"/>\n                  <y:Insets bottom="15" bottomF="15.0" left="15" leftF="15.0" right="15" rightF="15.0" top="15" topF="15.0"/>\n                  <y:BorderInsets bottom="1" bottomF="1.0" left="0" leftF="0.0" right="0" rightF="0.0" top="1" topF="1.0001736111111086"/>\n                </y:GroupNode>\n                <y:GroupNode>\n                  <y:Geometry height="50.0" width="50.0" x="0.0" y="60.0"/>\n                  <y:Fill color="#F5F5F5" transparent="false"/>\n                  <y:BorderStyle color="#000000" type="dashed" width="1.0"/>\n                  <y:NodeLabel alignment="right" autoSizePolicy="node_width" backgroundColor="#EBEBEB" borderDistance="0.0" fontFamily="Dialog" fontSize="15" fontStyle="plain" hasLineColor="false" height="21.4609375" horizontalTextPosition="center" iconTextGap="4" modelName="internal" modelPosition="t" textColor="#000000" verticalTextPosition="bottom" visible="true" width="65.201171875" x="-7.6005859375" y="0.0">Folder 1</y:NodeLabel>\n                  <y:Shape type="roundrectangle"/>\n                  <y:State closed="true" closedHeight="50.0" closedWidth="50.0" innerGraphDisplayEnabled="false"/>\n                  <y:Insets bottom="5" bottomF="5.0" left="5" leftF="5.0" right="5" rightF="5.0" top="5" topF="5.0"/>\n                  <y:BorderInsets bottom="0" bottomF="0.0" left="0" leftF="0.0" right="0" rightF="0.0" top="0" topF="0.0"/>\n                </y:GroupNode>\n              </y:Realizers>\n            </y:ProxyAutoBoundsNode>\n          </data>\n          <graph edgedefault="directed" id="n1::n0:">\n            <node id="n1::n0::n0">\n              <data key="d4"><![CDATA[CustomPropertyValue]]></data>\n              <data key="d6"/>\n              <data key="d7">\n                <y:ShapeNode>\n                  <y:Geometry height="30.0" width="30.0" x="50.0" y="-255.4611111111111"/>\n                  <y:Fill color="#FFCC00" transparent="false"/>\n                  <y:BorderStyle color="#000000" raised="false" type="line" width="1.0"/>\n                  <y:NodeLabel alignment="center" autoSizePolicy="content" fontFamily="Dialog" fontSize="12" fontStyle="plain" hasBackgroundColor="false" hasLineColor="false" height="17.96875" horizontalTextPosition="center" iconTextGap="4" modelName="custom" textColor="#000000" verticalTextPosition="bottom" visible="true" width="11.634765625" x="9.1826171875" y="6.015625">1<y:LabelModel>\n                      <y:SmartNodeLabelModel distance="4.0"/>\n                    </y:LabelModel>\n                    <y:ModelParameter>\n                      <y:SmartNodeLabelModelParameter labelRatioX="0.0" labelRatioY="0.0" nodeRatioX="0.0" nodeRatioY="0.0" offsetX="0.0" offsetY="0.0" upX="0.0" upY="-1.0"/>\n                    </y:ModelParameter>\n                  </y:NodeLabel>\n                  <y:Shape type="rectangle"/>\n                </y:ShapeNode>\n              </data>\n            </node>\n            <node id="n1::n0::n1">\n              <data key="d4"><![CDATA[CustomPropertyValue]]></data>\n              <data key="d6"/>\n              <data key="d7">\n                <y:ShapeNode>\n                  <y:Geometry height="30.0" width="30.0" x="0.0" y="-255.4611111111111"/>\n                  <y:Fill color="#FFCC00" transparent="false"/>\n                  <y:BorderStyle color="#000000" raised="false" type="line" width="1.0"/>\n                  <y:NodeLabel alignment="center" autoSizePolicy="content" fontFamily="Dialog" fontSize="12" fontStyle="plain" hasBackgroundColor="false" hasLineColor="false" height="17.96875" horizontalTextPosition="center" iconTextGap="4" modelName="custom" textColor="#000000" verticalTextPosition="bottom" visible="true" width="11.634765625" x="9.1826171875" y="6.015625">3<y:LabelModel>\n                      <y:SmartNodeLabelModel distance="4.0"/>\n                    </y:LabelModel>\n                    <y:ModelParameter>\n                      <y:SmartNodeLabelModelParameter labelRatioX="0.0" labelRatioY="0.0" nodeRatioX="0.0" nodeRatioY="0.0" offsetX="0.0" offsetY="0.0" upX="0.0" upY="-1.0"/>\n                    </y:ModelParameter>\n                  </y:NodeLabel>\n                  <y:Shape type="rectangle"/>\n                </y:ShapeNode>\n              </data>\n            </node>\n          </graph>\n        </node>\n        <node id="n1::n1" yfiles.foldertype="group">\n          <data key="d4"><![CDATA[CustomPropertyValue]]></data>\n          <data key="d5"/>\n          <data key="d6"/>\n          <data key="d7">\n            <y:ProxyAutoBoundsNode>\n              <y:Realizers active="0">\n                <y:GroupNode>\n                  <y:Geometry height="83.46111111111111" width="110.0" x="-15.0" y="-179.4611111111111"/>\n                  <y:Fill color="#F5F5F5" transparent="false"/>\n                  <y:BorderStyle color="#000000" type="dashed" width="1.0"/>\n                  <y:NodeLabel alignment="right" autoSizePolicy="node_width" backgroundColor="#EBEBEB" borderDistance="0.0" fontFamily="Dialog" fontSize="15" fontStyle="plain" hasLineColor="false" height="21.4609375" horizontalTextPosition="center" iconTextGap="4" modelName="internal" modelPosition="t" textColor="#000000" verticalTextPosition="bottom" visible="true" width="110.0" x="0.0" y="0.0">Group 2</y:NodeLabel>\n                  <y:Shape type="roundrectangle"/>\n                  <y:State closed="false" closedHeight="50.0" closedWidth="50.0" innerGraphDisplayEnabled="false"/>\n                  <y:Insets bottom="15" bottomF="15.0" left="15" leftF="15.0" right="15" rightF="15.0" top="15" topF="15.0"/>\n                  <y:BorderInsets bottom="1" bottomF="1.0" left="0" leftF="0.0" right="0" rightF="0.0" top="1" topF="1.0001736111111086"/>\n                </y:GroupNode>\n                <y:GroupNode>\n                  <y:Geometry height="50.0" width="50.0" x="0.0" y="60.0"/>\n                  <y:Fill color="#F5F5F5" transparent="false"/>\n                  <y:BorderStyle color="#000000" type="dashed" width="1.0"/>\n                  <y:NodeLabel alignment="right" autoSizePolicy="node_width" backgroundColor="#EBEBEB" borderDistance="0.0" fontFamily="Dialog" fontSize="15" fontStyle="plain" hasLineColor="false" height="21.4609375" horizontalTextPosition="center" iconTextGap="4" modelName="internal" modelPosition="t" textColor="#000000" verticalTextPosition="bottom" visible="true" width="65.201171875" x="-7.6005859375" y="0.0">Folder 2</y:NodeLabel>\n                  <y:Shape type="roundrectangle"/>\n                  <y:State closed="true" closedHeight="50.0" closedWidth="50.0" innerGraphDisplayEnabled="false"/>\n                  <y:Insets bottom="5" bottomF="5.0" left="5" leftF="5.0" right="5" rightF="5.0" top="5" topF="5.0"/>\n                  <y:BorderInsets bottom="0" bottomF="0.0" left="0" leftF="0.0" right="0" rightF="0.0" top="0" topF="0.0"/>\n                </y:GroupNode>\n              </y:Realizers>\n            </y:ProxyAutoBoundsNode>\n          </data>\n          <graph edgedefault="directed" id="n1::n1:">\n            <node id="n1::n1::n0">\n              <data key="d4"><![CDATA[CustomPropertyValue]]></data>\n              <data key="d6"/>\n              <data key="d7">\n                <y:ShapeNode>\n                  <y:Geometry height="30.0" width="30.0" x="0.0" y="-142.0"/>\n                  <y:Fill color="#FFCC00" transparent="false"/>\n                  <y:BorderStyle color="#000000" raised="false" type="line" width="1.0"/>\n                  <y:NodeLabel alignment="center" autoSizePolicy="content" fontFamily="Dialog" fontSize="12" fontStyle="plain" hasBackgroundColor="false" hasLineColor="false" height="17.96875" horizontalTextPosition="center" iconTextGap="4" modelName="custom" textColor="#000000" verticalTextPosition="bottom" visible="true" width="11.634765625" x="9.1826171875" y="6.015625">5<y:LabelModel>\n                      <y:SmartNodeLabelModel distance="4.0"/>\n                    </y:LabelModel>\n                    <y:ModelParameter>\n                      <y:SmartNodeLabelModelParameter labelRatioX="0.0" labelRatioY="0.0" nodeRatioX="0.0" nodeRatioY="0.0" offsetX="0.0" offsetY="0.0" upX="0.0" upY="-1.0"/>\n                    </y:ModelParameter>\n                  </y:NodeLabel>\n                  <y:Shape type="rectangle"/>\n                </y:ShapeNode>\n              </data>\n            </node>\n            <node id="n1::n1::n1">\n              <data key="d4"><![CDATA[CustomPropertyValue]]></data>\n              <data key="d6"/>\n              <data key="d7">\n                <y:ShapeNode>\n                  <y:Geometry height="30.0" width="30.0" x="50.0" y="-142.0"/>\n                  <y:Fill color="#FFCC00" transparent="false"/>\n                  <y:BorderStyle color="#000000" raised="false" type="line" width="1.0"/>\n                  <y:NodeLabel alignment="center" autoSizePolicy="content" fontFamily="Dialog" fontSize="12" fontStyle="plain" hasBackgroundColor="false" hasLineColor="false" height="17.96875" horizontalTextPosition="center" iconTextGap="4" modelName="custom" textColor="#000000" verticalTextPosition="bottom" visible="true" width="11.634765625" x="9.1826171875" y="6.015625">6<y:LabelModel>\n                      <y:SmartNodeLabelModel distance="4.0"/>\n                    </y:LabelModel>\n                    <y:ModelParameter>\n                      <y:SmartNodeLabelModelParameter labelRatioX="0.0" labelRatioY="0.0" nodeRatioX="0.0" nodeRatioY="0.0" offsetX="0.0" offsetY="0.0" upX="0.0" upY="-1.0"/>\n                    </y:ModelParameter>\n                  </y:NodeLabel>\n                  <y:Shape type="rectangle"/>\n                </y:ShapeNode>\n              </data>\n            </node>\n          </graph>\n        </node>\n      </graph>\n    </node>\n    <node id="n2">\n      <data key="d4"><![CDATA[CustomPropertyValue]]></data>\n      <data key="d6"/>\n      <data key="d7">\n        <y:ShapeNode>\n          <y:Geometry height="30.0" width="30.0" x="125.0" y="-142.0"/>\n          <y:Fill color="#FFCC00" transparent="false"/>\n          <y:BorderStyle color="#000000" raised="false" type="line" width="1.0"/>\n          <y:NodeLabel alignment="center" autoSizePolicy="content" fontFamily="Dialog" fontSize="12" fontStyle="plain" hasBackgroundColor="false" hasLineColor="false" height="17.96875" horizontalTextPosition="center" iconTextGap="4" modelName="custom" textColor="#000000" verticalTextPosition="bottom" visible="true" width="11.634765625" x="9.1826171875" y="6.015625">9<y:LabelModel>\n              <y:SmartNodeLabelModel distance="4.0"/>\n            </y:LabelModel>\n            <y:ModelParameter>\n              <y:SmartNodeLabelModelParameter labelRatioX="0.0" labelRatioY="0.0" nodeRatioX="0.0" nodeRatioY="0.0" offsetX="0.0" offsetY="0.0" upX="0.0" upY="-1.0"/>\n            </y:ModelParameter>\n          </y:NodeLabel>\n          <y:Shape type="rectangle"/>\n        </y:ShapeNode>\n      </data>\n    </node>\n    <edge id="n1::n1::e0" source="n1::n1::n0" target="n1::n1::n1">\n      <data key="d10"/>\n      <data key="d11">\n        <y:PolyLineEdge>\n          <y:Path sx="15.0" sy="-0.0" tx="-15.0" ty="-0.0"/>\n          <y:LineStyle color="#000000" type="line" width="1.0"/>\n          <y:Arrows source="none" target="standard"/>\n          <y:BendStyle smoothed="false"/>\n        </y:PolyLineEdge>\n      </data>\n    </edge>\n    <edge id="n1::n0::e0" source="n1::n0::n1" target="n1::n0::n0">\n      <data key="d10"/>\n      <data key="d11">\n        <y:PolyLineEdge>\n          <y:Path sx="15.0" sy="-0.0" tx="-15.0" ty="-0.0"/>\n          <y:LineStyle color="#000000" type="line" width="1.0"/>\n          <y:Arrows source="none" target="standard"/>\n          <y:BendStyle smoothed="false"/>\n        </y:PolyLineEdge>\n      </data>\n    </edge>\n    <edge id="e0" source="n1::n0::n0" target="n0">\n      <data key="d10"/>\n      <data key="d11">\n        <y:PolyLineEdge>\n          <y:Path sx="15.0" sy="-0.0" tx="-15.0" ty="-0.0"/>\n          <y:LineStyle color="#000000" type="line" width="1.0"/>\n          <y:Arrows source="none" target="standard"/>\n          <y:BendStyle smoothed="false"/>\n        </y:PolyLineEdge>\n      </data>\n    </edge>\n    <edge id="e1" source="n1::n1::n1" target="n2">\n      <data key="d10"/>\n      <data key="d11">\n        <y:PolyLineEdge>\n          <y:Path sx="15.0" sy="-0.0" tx="-15.0" ty="-0.0"/>\n          <y:LineStyle color="#000000" type="line" width="1.0"/>\n          <y:Arrows source="none" target="standard"/>\n          <y:BendStyle smoothed="false"/>\n        </y:PolyLineEdge>\n      </data>\n    </edge>\n  </graph>\n  <data key="d8">\n    <y:Resources/>\n  </data>\n</graphml>\n'
        fh = io.BytesIO(data.encode('UTF-8'))
        G = nx.read_graphml(fh)
        data = [x for _, x in G.nodes(data=True)]
        assert len(data) == 9
        for node_data in data:
            assert node_data['CustomProperty'] != ''

    def test_long_attribute_type(self):
        s = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <key attr.name="cudfversion" attr.type="long" for="node" id="d6" />\n  <graph edgedefault="directed">\n    <node id="n1">\n      <data key="d6">4284</data>\n    </node>\n  </graph>\n</graphml>'
        fh = io.BytesIO(s.encode('UTF-8'))
        G = nx.read_graphml(fh)
        expected = [('n1', {'cudfversion': 4284})]
        assert sorted(G.nodes(data=True)) == expected
        fh.seek(0)
        H = nx.parse_graphml(s)
        assert sorted(H.nodes(data=True)) == expected