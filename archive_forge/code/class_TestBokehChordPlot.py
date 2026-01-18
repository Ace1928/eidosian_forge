import numpy as np
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges, NodesOnly, Patches
from bokeh.models.mappers import CategoricalColorMapper, LinearColorMapper
from holoviews.core.data import Dataset
from holoviews.element import Chord, Graph, Nodes, TriMesh, VLine, circular_layout
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
class TestBokehChordPlot(TestBokehPlot):

    def setUp(self):
        super().setUp()
        self.edges = [(0, 1, 1), (0, 2, 2), (1, 2, 3)]
        self.nodes = Dataset([(0, 'A'), (1, 'B'), (2, 'C')], 'index', 'Label')
        self.chord = Chord((self.edges, self.nodes))

    def test_chord_draw_order(self):
        plot = bokeh_renderer.get_plot(self.chord)
        renderers = plot.state.renderers
        graph_renderer = plot.handles['glyph_renderer']
        arc_renderer = plot.handles['multi_line_2_glyph_renderer']
        self.assertTrue(renderers.index(arc_renderer) < renderers.index(graph_renderer))

    def test_chord_label_draw_order(self):
        g = self.chord.opts(labels='Label')
        plot = bokeh_renderer.get_plot(g)
        renderers = plot.state.renderers
        graph_renderer = plot.handles['glyph_renderer']
        label_renderer = plot.handles['text_1_glyph_renderer']
        self.assertTrue(renderers.index(graph_renderer) < renderers.index(label_renderer))

    def test_chord_nodes_label_text(self):
        g = self.chord.opts(label_index='Label')
        plot = bokeh_renderer.get_plot(g)
        source = plot.handles['text_1_source']
        self.assertEqual(source.data['text'], ['A', 'B', 'C'])

    def test_chord_nodes_labels_mapping(self):
        g = self.chord.opts(labels='Label')
        plot = bokeh_renderer.get_plot(g)
        source = plot.handles['text_1_source']
        self.assertEqual(source.data['text'], ['A', 'B', 'C'])

    def test_chord_nodes_categorically_colormapped(self):
        g = self.chord.opts(color_index='Label', label_index='Label', cmap=['#FFFFFF', '#888888', '#000000'])
        plot = bokeh_renderer.get_plot(g)
        cmapper = plot.handles['color_mapper']
        source = plot.handles['scatter_1_source']
        arc_source = plot.handles['multi_line_2_source']
        glyph = plot.handles['scatter_1_glyph']
        self.assertIsInstance(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.factors, ['A', 'B', 'C'])
        self.assertEqual(cmapper.palette, ['#FFFFFF', '#888888', '#000000'])
        self.assertEqual(source.data['Label'], np.array(['A', 'B', 'C']))
        self.assertEqual(arc_source.data['Label'], np.array(['A', 'B', 'C']))
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'Label', 'transform': cmapper})

    def test_chord_nodes_style_map_node_color_colormapped(self):
        g = self.chord.opts(labels='Label', node_color='Label', cmap=['#FFFFFF', '#888888', '#000000'])
        plot = bokeh_renderer.get_plot(g)
        cmapper = plot.handles['node_color_color_mapper']
        source = plot.handles['scatter_1_source']
        arc_source = plot.handles['multi_line_2_source']
        glyph = plot.handles['scatter_1_glyph']
        arc_glyph = plot.handles['multi_line_2_glyph']
        self.assertIsInstance(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.factors, ['A', 'B', 'C'])
        self.assertEqual(cmapper.palette, ['#FFFFFF', '#888888', '#000000'])
        self.assertEqual(source.data['Label'], np.array(['A', 'B', 'C']))
        self.assertEqual(arc_source.data['Label'], np.array(['A', 'B', 'C']))
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'node_color', 'transform': cmapper})
        self.assertEqual(property_to_dict(arc_glyph.line_color), {'field': 'node_color', 'transform': cmapper})

    def test_chord_edges_categorically_colormapped(self):
        g = self.chord.opts(edge_color_index='start', edge_cmap=['#FFFFFF', '#000000'])
        plot = bokeh_renderer.get_plot(g)
        cmapper = plot.handles['edge_colormapper']
        edge_source = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        self.assertIsInstance(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.palette, ['#FFFFFF', '#000000', '#FFFFFF'])
        self.assertEqual(cmapper.factors, ['0', '1', '2'])
        self.assertEqual(edge_source.data['start_str__'], ['0', '0', '1'])
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'start_str__', 'transform': cmapper})

    def test_chord_edge_color_style_mapping(self):
        g = self.chord.opts(edge_color=dim('start').astype(str), edge_cmap=['#FFFFFF', '#000000'])
        plot = bokeh_renderer.get_plot(g)
        cmapper = plot.handles['edge_color_color_mapper']
        edge_source = plot.handles['multi_line_1_source']
        glyph = plot.handles['multi_line_1_glyph']
        self.assertIsInstance(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.palette, ['#FFFFFF', '#000000', '#FFFFFF'])
        self.assertEqual(cmapper.factors, ['0', '1', '2'])
        self.assertEqual(edge_source.data['edge_color'], np.array(['0', '0', '1']))
        self.assertEqual(property_to_dict(glyph.line_color), {'field': 'edge_color', 'transform': cmapper})