import itertools
import time
from xml.etree.ElementTree import (
import networkx as nx
from networkx.utils import open_file
class GEXFWriter(GEXF):

    def __init__(self, graph=None, encoding='utf-8', prettyprint=True, version='1.2draft'):
        self.construct_types()
        self.prettyprint = prettyprint
        self.encoding = encoding
        self.set_version(version)
        self.xml = Element('gexf', {'xmlns': self.NS_GEXF, 'xmlns:xsi': self.NS_XSI, 'xsi:schemaLocation': self.SCHEMALOCATION, 'version': self.VERSION})
        meta_element = Element('meta')
        subelement_text = f'NetworkX {nx.__version__}'
        SubElement(meta_element, 'creator').text = subelement_text
        meta_element.set('lastmodifieddate', time.strftime('%Y-%m-%d'))
        self.xml.append(meta_element)
        register_namespace('viz', self.NS_VIZ)
        self.edge_id = itertools.count()
        self.attr_id = itertools.count()
        self.all_edge_ids = set()
        self.attr = {}
        self.attr['node'] = {}
        self.attr['edge'] = {}
        self.attr['node']['dynamic'] = {}
        self.attr['node']['static'] = {}
        self.attr['edge']['dynamic'] = {}
        self.attr['edge']['static'] = {}
        if graph is not None:
            self.add_graph(graph)

    def __str__(self):
        if self.prettyprint:
            self.indent(self.xml)
        s = tostring(self.xml).decode(self.encoding)
        return s

    def add_graph(self, G):
        for u, v, dd in G.edges(data=True):
            eid = dd.get('id')
            if eid is not None:
                self.all_edge_ids.add(str(eid))
        if G.graph.get('mode') == 'dynamic':
            mode = 'dynamic'
        else:
            mode = 'static'
        if G.is_directed():
            default = 'directed'
        else:
            default = 'undirected'
        name = G.graph.get('name', '')
        graph_element = Element('graph', defaultedgetype=default, mode=mode, name=name)
        self.graph_element = graph_element
        self.add_nodes(G, graph_element)
        self.add_edges(G, graph_element)
        self.xml.append(graph_element)

    def add_nodes(self, G, graph_element):
        nodes_element = Element('nodes')
        for node, data in G.nodes(data=True):
            node_data = data.copy()
            node_id = str(node_data.pop('id', node))
            kw = {'id': node_id}
            label = str(node_data.pop('label', node))
            kw['label'] = label
            try:
                pid = node_data.pop('pid')
                kw['pid'] = str(pid)
            except KeyError:
                pass
            try:
                start = node_data.pop('start')
                kw['start'] = str(start)
                self.alter_graph_mode_timeformat(start)
            except KeyError:
                pass
            try:
                end = node_data.pop('end')
                kw['end'] = str(end)
                self.alter_graph_mode_timeformat(end)
            except KeyError:
                pass
            node_element = Element('node', **kw)
            default = G.graph.get('node_default', {})
            node_data = self.add_parents(node_element, node_data)
            if self.VERSION == '1.1':
                node_data = self.add_slices(node_element, node_data)
            else:
                node_data = self.add_spells(node_element, node_data)
            node_data = self.add_viz(node_element, node_data)
            node_data = self.add_attributes('node', node_element, node_data, default)
            nodes_element.append(node_element)
        graph_element.append(nodes_element)

    def add_edges(self, G, graph_element):

        def edge_key_data(G):
            if G.is_multigraph():
                for u, v, key, data in G.edges(data=True, keys=True):
                    edge_data = data.copy()
                    edge_data.update(key=key)
                    edge_id = edge_data.pop('id', None)
                    if edge_id is None:
                        edge_id = next(self.edge_id)
                        while str(edge_id) in self.all_edge_ids:
                            edge_id = next(self.edge_id)
                        self.all_edge_ids.add(str(edge_id))
                    yield (u, v, edge_id, edge_data)
            else:
                for u, v, data in G.edges(data=True):
                    edge_data = data.copy()
                    edge_id = edge_data.pop('id', None)
                    if edge_id is None:
                        edge_id = next(self.edge_id)
                        while str(edge_id) in self.all_edge_ids:
                            edge_id = next(self.edge_id)
                        self.all_edge_ids.add(str(edge_id))
                    yield (u, v, edge_id, edge_data)
        edges_element = Element('edges')
        for u, v, key, edge_data in edge_key_data(G):
            kw = {'id': str(key)}
            try:
                edge_label = edge_data.pop('label')
                kw['label'] = str(edge_label)
            except KeyError:
                pass
            try:
                edge_weight = edge_data.pop('weight')
                kw['weight'] = str(edge_weight)
            except KeyError:
                pass
            try:
                edge_type = edge_data.pop('type')
                kw['type'] = str(edge_type)
            except KeyError:
                pass
            try:
                start = edge_data.pop('start')
                kw['start'] = str(start)
                self.alter_graph_mode_timeformat(start)
            except KeyError:
                pass
            try:
                end = edge_data.pop('end')
                kw['end'] = str(end)
                self.alter_graph_mode_timeformat(end)
            except KeyError:
                pass
            source_id = str(G.nodes[u].get('id', u))
            target_id = str(G.nodes[v].get('id', v))
            edge_element = Element('edge', source=source_id, target=target_id, **kw)
            default = G.graph.get('edge_default', {})
            if self.VERSION == '1.1':
                edge_data = self.add_slices(edge_element, edge_data)
            else:
                edge_data = self.add_spells(edge_element, edge_data)
            edge_data = self.add_viz(edge_element, edge_data)
            edge_data = self.add_attributes('edge', edge_element, edge_data, default)
            edges_element.append(edge_element)
        graph_element.append(edges_element)

    def add_attributes(self, node_or_edge, xml_obj, data, default):
        attvalues = Element('attvalues')
        if len(data) == 0:
            return data
        mode = 'static'
        for k, v in data.items():
            if k == 'key':
                k = 'networkx_key'
            val_type = type(v)
            if val_type not in self.xml_type:
                raise TypeError(f'attribute value type is not allowed: {val_type}')
            if isinstance(v, list):
                for val, start, end in v:
                    val_type = type(val)
                    if start is not None or end is not None:
                        mode = 'dynamic'
                        self.alter_graph_mode_timeformat(start)
                        self.alter_graph_mode_timeformat(end)
                        break
                attr_id = self.get_attr_id(str(k), self.xml_type[val_type], node_or_edge, default, mode)
                for val, start, end in v:
                    e = Element('attvalue')
                    e.attrib['for'] = attr_id
                    e.attrib['value'] = str(val)
                    if val_type == float:
                        if e.attrib['value'] == 'inf':
                            e.attrib['value'] = 'INF'
                        elif e.attrib['value'] == 'nan':
                            e.attrib['value'] = 'NaN'
                        elif e.attrib['value'] == '-inf':
                            e.attrib['value'] = '-INF'
                    if start is not None:
                        e.attrib['start'] = str(start)
                    if end is not None:
                        e.attrib['end'] = str(end)
                    attvalues.append(e)
            else:
                mode = 'static'
                attr_id = self.get_attr_id(str(k), self.xml_type[val_type], node_or_edge, default, mode)
                e = Element('attvalue')
                e.attrib['for'] = attr_id
                if isinstance(v, bool):
                    e.attrib['value'] = str(v).lower()
                else:
                    e.attrib['value'] = str(v)
                    if val_type == float:
                        if e.attrib['value'] == 'inf':
                            e.attrib['value'] = 'INF'
                        elif e.attrib['value'] == 'nan':
                            e.attrib['value'] = 'NaN'
                        elif e.attrib['value'] == '-inf':
                            e.attrib['value'] = '-INF'
                attvalues.append(e)
        xml_obj.append(attvalues)
        return data

    def get_attr_id(self, title, attr_type, edge_or_node, default, mode):
        try:
            return self.attr[edge_or_node][mode][title]
        except KeyError:
            new_id = str(next(self.attr_id))
            self.attr[edge_or_node][mode][title] = new_id
            attr_kwargs = {'id': new_id, 'title': title, 'type': attr_type}
            attribute = Element('attribute', **attr_kwargs)
            default_title = default.get(title)
            if default_title is not None:
                default_element = Element('default')
                default_element.text = str(default_title)
                attribute.append(default_element)
            attributes_element = None
            for a in self.graph_element.findall('attributes'):
                a_class = a.get('class')
                a_mode = a.get('mode', 'static')
                if a_class == edge_or_node and a_mode == mode:
                    attributes_element = a
            if attributes_element is None:
                attr_kwargs = {'mode': mode, 'class': edge_or_node}
                attributes_element = Element('attributes', **attr_kwargs)
                self.graph_element.insert(0, attributes_element)
            attributes_element.append(attribute)
        return new_id

    def add_viz(self, element, node_data):
        viz = node_data.pop('viz', False)
        if viz:
            color = viz.get('color')
            if color is not None:
                if self.VERSION == '1.1':
                    e = Element(f'{{{self.NS_VIZ}}}color', r=str(color.get('r')), g=str(color.get('g')), b=str(color.get('b')))
                else:
                    e = Element(f'{{{self.NS_VIZ}}}color', r=str(color.get('r')), g=str(color.get('g')), b=str(color.get('b')), a=str(color.get('a', 1.0)))
                element.append(e)
            size = viz.get('size')
            if size is not None:
                e = Element(f'{{{self.NS_VIZ}}}size', value=str(size))
                element.append(e)
            thickness = viz.get('thickness')
            if thickness is not None:
                e = Element(f'{{{self.NS_VIZ}}}thickness', value=str(thickness))
                element.append(e)
            shape = viz.get('shape')
            if shape is not None:
                if shape.startswith('http'):
                    e = Element(f'{{{self.NS_VIZ}}}shape', value='image', uri=str(shape))
                else:
                    e = Element(f'{{{self.NS_VIZ}}}shape', value=str(shape))
                element.append(e)
            position = viz.get('position')
            if position is not None:
                e = Element(f'{{{self.NS_VIZ}}}position', x=str(position.get('x')), y=str(position.get('y')), z=str(position.get('z')))
                element.append(e)
        return node_data

    def add_parents(self, node_element, node_data):
        parents = node_data.pop('parents', False)
        if parents:
            parents_element = Element('parents')
            for p in parents:
                e = Element('parent')
                e.attrib['for'] = str(p)
                parents_element.append(e)
            node_element.append(parents_element)
        return node_data

    def add_slices(self, node_or_edge_element, node_or_edge_data):
        slices = node_or_edge_data.pop('slices', False)
        if slices:
            slices_element = Element('slices')
            for start, end in slices:
                e = Element('slice', start=str(start), end=str(end))
                slices_element.append(e)
            node_or_edge_element.append(slices_element)
        return node_or_edge_data

    def add_spells(self, node_or_edge_element, node_or_edge_data):
        spells = node_or_edge_data.pop('spells', False)
        if spells:
            spells_element = Element('spells')
            for start, end in spells:
                e = Element('spell')
                if start is not None:
                    e.attrib['start'] = str(start)
                    self.alter_graph_mode_timeformat(start)
                if end is not None:
                    e.attrib['end'] = str(end)
                    self.alter_graph_mode_timeformat(end)
                spells_element.append(e)
            node_or_edge_element.append(spells_element)
        return node_or_edge_data

    def alter_graph_mode_timeformat(self, start_or_end):
        if self.graph_element.get('mode') == 'static':
            if start_or_end is not None:
                if isinstance(start_or_end, str):
                    timeformat = 'date'
                elif isinstance(start_or_end, float):
                    timeformat = 'double'
                elif isinstance(start_or_end, int):
                    timeformat = 'long'
                else:
                    raise nx.NetworkXError('timeformat should be of the type int, float or str')
                self.graph_element.set('timeformat', timeformat)
                self.graph_element.set('mode', 'dynamic')

    def write(self, fh):
        if self.prettyprint:
            self.indent(self.xml)
        document = ElementTree(self.xml)
        document.write(fh, encoding=self.encoding, xml_declaration=True)

    def indent(self, elem, level=0):
        i = '\n' + '  ' * level
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + '  '
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        elif level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i