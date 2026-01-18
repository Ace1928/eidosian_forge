import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
class BaseGraphML:

    @classmethod
    def setup_class(cls):
        cls.simple_directed_data = '<?xml version="1.0" encoding="UTF-8"?>\n<!-- This file was written by the JAVA GraphML Library.-->\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <graph id="G" edgedefault="directed">\n    <node id="n0"/>\n    <node id="n1"/>\n    <node id="n2"/>\n    <node id="n3"/>\n    <node id="n4"/>\n    <node id="n5"/>\n    <node id="n6"/>\n    <node id="n7"/>\n    <node id="n8"/>\n    <node id="n9"/>\n    <node id="n10"/>\n    <edge id="foo" source="n0" target="n2"/>\n    <edge source="n1" target="n2"/>\n    <edge source="n2" target="n3"/>\n    <edge source="n3" target="n5"/>\n    <edge source="n3" target="n4"/>\n    <edge source="n4" target="n6"/>\n    <edge source="n6" target="n5"/>\n    <edge source="n5" target="n7"/>\n    <edge source="n6" target="n8"/>\n    <edge source="n8" target="n7"/>\n    <edge source="n8" target="n9"/>\n  </graph>\n</graphml>'
        cls.simple_directed_graph = nx.DiGraph()
        cls.simple_directed_graph.add_node('n10')
        cls.simple_directed_graph.add_edge('n0', 'n2', id='foo')
        cls.simple_directed_graph.add_edge('n0', 'n2')
        cls.simple_directed_graph.add_edges_from([('n1', 'n2'), ('n2', 'n3'), ('n3', 'n5'), ('n3', 'n4'), ('n4', 'n6'), ('n6', 'n5'), ('n5', 'n7'), ('n6', 'n8'), ('n8', 'n7'), ('n8', 'n9')])
        cls.simple_directed_fh = io.BytesIO(cls.simple_directed_data.encode('UTF-8'))
        cls.attribute_data = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n      xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n        http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <key id="d0" for="node" attr.name="color" attr.type="string">\n    <default>yellow</default>\n  </key>\n  <key id="d1" for="edge" attr.name="weight" attr.type="double"/>\n  <graph id="G" edgedefault="directed">\n    <node id="n0">\n      <data key="d0">green</data>\n    </node>\n    <node id="n1"/>\n    <node id="n2">\n      <data key="d0">blue</data>\n    </node>\n    <node id="n3">\n      <data key="d0">red</data>\n    </node>\n    <node id="n4"/>\n    <node id="n5">\n      <data key="d0">turquoise</data>\n    </node>\n    <edge id="e0" source="n0" target="n2">\n      <data key="d1">1.0</data>\n    </edge>\n    <edge id="e1" source="n0" target="n1">\n      <data key="d1">1.0</data>\n    </edge>\n    <edge id="e2" source="n1" target="n3">\n      <data key="d1">2.0</data>\n    </edge>\n    <edge id="e3" source="n3" target="n2"/>\n    <edge id="e4" source="n2" target="n4"/>\n    <edge id="e5" source="n3" target="n5"/>\n    <edge id="e6" source="n5" target="n4">\n      <data key="d1">1.1</data>\n    </edge>\n  </graph>\n</graphml>\n'
        cls.attribute_graph = nx.DiGraph(id='G')
        cls.attribute_graph.graph['node_default'] = {'color': 'yellow'}
        cls.attribute_graph.add_node('n0', color='green')
        cls.attribute_graph.add_node('n2', color='blue')
        cls.attribute_graph.add_node('n3', color='red')
        cls.attribute_graph.add_node('n4')
        cls.attribute_graph.add_node('n5', color='turquoise')
        cls.attribute_graph.add_edge('n0', 'n2', id='e0', weight=1.0)
        cls.attribute_graph.add_edge('n0', 'n1', id='e1', weight=1.0)
        cls.attribute_graph.add_edge('n1', 'n3', id='e2', weight=2.0)
        cls.attribute_graph.add_edge('n3', 'n2', id='e3')
        cls.attribute_graph.add_edge('n2', 'n4', id='e4')
        cls.attribute_graph.add_edge('n3', 'n5', id='e5')
        cls.attribute_graph.add_edge('n5', 'n4', id='e6', weight=1.1)
        cls.attribute_fh = io.BytesIO(cls.attribute_data.encode('UTF-8'))
        cls.node_attribute_default_data = '<?xml version="1.0" encoding="UTF-8"?>\n        <graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n              xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n              xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n                http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n          <key id="d0" for="node" attr.name="boolean_attribute" attr.type="boolean"><default>false</default></key>\n          <key id="d1" for="node" attr.name="int_attribute" attr.type="int"><default>0</default></key>\n          <key id="d2" for="node" attr.name="long_attribute" attr.type="long"><default>0</default></key>\n          <key id="d3" for="node" attr.name="float_attribute" attr.type="float"><default>0.0</default></key>\n          <key id="d4" for="node" attr.name="double_attribute" attr.type="double"><default>0.0</default></key>\n          <key id="d5" for="node" attr.name="string_attribute" attr.type="string"><default>Foo</default></key>\n          <graph id="G" edgedefault="directed">\n            <node id="n0"/>\n            <node id="n1"/>\n            <edge id="e0" source="n0" target="n1"/>\n          </graph>\n        </graphml>\n        '
        cls.node_attribute_default_graph = nx.DiGraph(id='G')
        cls.node_attribute_default_graph.graph['node_default'] = {'boolean_attribute': False, 'int_attribute': 0, 'long_attribute': 0, 'float_attribute': 0.0, 'double_attribute': 0.0, 'string_attribute': 'Foo'}
        cls.node_attribute_default_graph.add_node('n0')
        cls.node_attribute_default_graph.add_node('n1')
        cls.node_attribute_default_graph.add_edge('n0', 'n1', id='e0')
        cls.node_attribute_default_fh = io.BytesIO(cls.node_attribute_default_data.encode('UTF-8'))
        cls.attribute_named_key_ids_data = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <key id="edge_prop" for="edge" attr.name="edge_prop" attr.type="string"/>\n  <key id="prop2" for="node" attr.name="prop2" attr.type="string"/>\n  <key id="prop1" for="node" attr.name="prop1" attr.type="string"/>\n  <graph edgedefault="directed">\n    <node id="0">\n      <data key="prop1">val1</data>\n      <data key="prop2">val2</data>\n    </node>\n    <node id="1">\n      <data key="prop1">val_one</data>\n      <data key="prop2">val2</data>\n    </node>\n    <edge source="0" target="1">\n      <data key="edge_prop">edge_value</data>\n    </edge>\n  </graph>\n</graphml>\n'
        cls.attribute_named_key_ids_graph = nx.DiGraph()
        cls.attribute_named_key_ids_graph.add_node('0', prop1='val1', prop2='val2')
        cls.attribute_named_key_ids_graph.add_node('1', prop1='val_one', prop2='val2')
        cls.attribute_named_key_ids_graph.add_edge('0', '1', edge_prop='edge_value')
        fh = io.BytesIO(cls.attribute_named_key_ids_data.encode('UTF-8'))
        cls.attribute_named_key_ids_fh = fh
        cls.attribute_numeric_type_data = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <key attr.name="weight" attr.type="double" for="node" id="d1" />\n  <key attr.name="weight" attr.type="double" for="edge" id="d0" />\n  <graph edgedefault="directed">\n    <node id="n0">\n      <data key="d1">1</data>\n    </node>\n    <node id="n1">\n      <data key="d1">2.0</data>\n    </node>\n    <edge source="n0" target="n1">\n      <data key="d0">1</data>\n    </edge>\n    <edge source="n1" target="n0">\n      <data key="d0">k</data>\n    </edge>\n    <edge source="n1" target="n1">\n      <data key="d0">1.0</data>\n    </edge>\n  </graph>\n</graphml>\n'
        cls.attribute_numeric_type_graph = nx.DiGraph()
        cls.attribute_numeric_type_graph.add_node('n0', weight=1)
        cls.attribute_numeric_type_graph.add_node('n1', weight=2.0)
        cls.attribute_numeric_type_graph.add_edge('n0', 'n1', weight=1)
        cls.attribute_numeric_type_graph.add_edge('n1', 'n1', weight=1.0)
        fh = io.BytesIO(cls.attribute_numeric_type_data.encode('UTF-8'))
        cls.attribute_numeric_type_fh = fh
        cls.simple_undirected_data = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <graph id="G">\n    <node id="n0"/>\n    <node id="n1"/>\n    <node id="n2"/>\n    <node id="n10"/>\n    <edge id="foo" source="n0" target="n2"/>\n    <edge source="n1" target="n2"/>\n    <edge source="n2" target="n3"/>\n  </graph>\n</graphml>'
        cls.simple_undirected_graph = nx.Graph()
        cls.simple_undirected_graph.add_node('n10')
        cls.simple_undirected_graph.add_edge('n0', 'n2', id='foo')
        cls.simple_undirected_graph.add_edges_from([('n1', 'n2'), ('n2', 'n3')])
        fh = io.BytesIO(cls.simple_undirected_data.encode('UTF-8'))
        cls.simple_undirected_fh = fh
        cls.undirected_multigraph_data = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <graph id="G">\n    <node id="n0"/>\n    <node id="n1"/>\n    <node id="n2"/>\n    <node id="n10"/>\n    <edge id="e0" source="n0" target="n2"/>\n    <edge id="e1" source="n1" target="n2"/>\n    <edge id="e2" source="n2" target="n1"/>\n  </graph>\n</graphml>'
        cls.undirected_multigraph = nx.MultiGraph()
        cls.undirected_multigraph.add_node('n10')
        cls.undirected_multigraph.add_edge('n0', 'n2', id='e0')
        cls.undirected_multigraph.add_edge('n1', 'n2', id='e1')
        cls.undirected_multigraph.add_edge('n2', 'n1', id='e2')
        fh = io.BytesIO(cls.undirected_multigraph_data.encode('UTF-8'))
        cls.undirected_multigraph_fh = fh
        cls.undirected_multigraph_no_multiedge_data = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <graph id="G">\n    <node id="n0"/>\n    <node id="n1"/>\n    <node id="n2"/>\n    <node id="n10"/>\n    <edge id="e0" source="n0" target="n2"/>\n    <edge id="e1" source="n1" target="n2"/>\n    <edge id="e2" source="n2" target="n3"/>\n  </graph>\n</graphml>'
        cls.undirected_multigraph_no_multiedge = nx.MultiGraph()
        cls.undirected_multigraph_no_multiedge.add_node('n10')
        cls.undirected_multigraph_no_multiedge.add_edge('n0', 'n2', id='e0')
        cls.undirected_multigraph_no_multiedge.add_edge('n1', 'n2', id='e1')
        cls.undirected_multigraph_no_multiedge.add_edge('n2', 'n3', id='e2')
        fh = io.BytesIO(cls.undirected_multigraph_no_multiedge_data.encode('UTF-8'))
        cls.undirected_multigraph_no_multiedge_fh = fh
        cls.multigraph_only_ids_for_multiedges_data = '<?xml version="1.0" encoding="UTF-8"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <graph id="G">\n    <node id="n0"/>\n    <node id="n1"/>\n    <node id="n2"/>\n    <node id="n10"/>\n    <edge source="n0" target="n2"/>\n    <edge id="e1" source="n1" target="n2"/>\n    <edge id="e2" source="n2" target="n1"/>\n  </graph>\n</graphml>'
        cls.multigraph_only_ids_for_multiedges = nx.MultiGraph()
        cls.multigraph_only_ids_for_multiedges.add_node('n10')
        cls.multigraph_only_ids_for_multiedges.add_edge('n0', 'n2')
        cls.multigraph_only_ids_for_multiedges.add_edge('n1', 'n2', id='e1')
        cls.multigraph_only_ids_for_multiedges.add_edge('n2', 'n1', id='e2')
        fh = io.BytesIO(cls.multigraph_only_ids_for_multiedges_data.encode('UTF-8'))
        cls.multigraph_only_ids_for_multiedges_fh = fh