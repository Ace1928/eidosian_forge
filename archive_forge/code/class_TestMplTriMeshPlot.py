import numpy as np
import pytest
from matplotlib.collections import LineCollection, PolyCollection
from packaging.version import Version
from holoviews.core.data import Dataset
from holoviews.core.options import AbbreviatedException, Cycle
from holoviews.core.spaces import HoloMap
from holoviews.element import Chord, Graph, Nodes, TriMesh, circular_layout
from holoviews.util.transform import dim
from .test_plot import TestMPLPlot, mpl_renderer
class TestMplTriMeshPlot(TestMPLPlot):

    def setUp(self):
        super().setUp()
        self.nodes = [(0, 0, 0), (0.5, 1, 1), (1.0, 0, 2), (1.5, 1, 3)]
        self.simplices = [(0, 1, 2, 0), (1, 2, 3, 1)]
        self.trimesh = TriMesh((self.simplices, self.nodes))
        self.trimesh_weighted = TriMesh((self.simplices, self.nodes), vdims='weight')

    def test_plot_simple_trimesh(self):
        plot = mpl_renderer.get_plot(self.trimesh)
        nodes = plot.handles['nodes']
        edges = plot.handles['edges']
        self.assertIsInstance(edges, LineCollection)
        self.assertEqual(np.asarray(nodes.get_offsets()), self.trimesh.nodes.array([0, 1]))
        self.assertEqual([p.vertices for p in edges.get_paths()], [p.array() for p in self.trimesh._split_edgepaths.split()])

    def test_plot_simple_trimesh_filled(self):
        plot = mpl_renderer.get_plot(self.trimesh.opts(filled=True))
        nodes = plot.handles['nodes']
        edges = plot.handles['edges']
        self.assertIsInstance(edges, PolyCollection)
        self.assertEqual(np.asarray(nodes.get_offsets()), self.trimesh.nodes.array([0, 1]))
        paths = self.trimesh._split_edgepaths.split(datatype='array')
        self.assertEqual([p.vertices[:4] for p in edges.get_paths()], paths)

    def test_plot_trimesh_colored_edges(self):
        opts = dict(edge_color_index='weight', edge_cmap='Greys')
        plot = mpl_renderer.get_plot(self.trimesh_weighted.opts(**opts))
        edges = plot.handles['edges']
        colors = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
        self.assertEqual(edges.get_edgecolors(), colors)

    def test_plot_trimesh_categorically_colored_edges(self):
        opts = dict(edge_color_index='node1', edge_color=Cycle('Set1'))
        plot = mpl_renderer.get_plot(self.trimesh_weighted.opts(**opts))
        edges = plot.handles['edges']
        colors = np.array([[0.894118, 0.101961, 0.109804, 1.0], [0.215686, 0.494118, 0.721569, 1.0]])
        self.assertEqual(edges.get_edgecolors(), colors)

    def test_plot_trimesh_categorically_colored_edges_filled(self):
        opts = dict(edge_color_index='node1', filled=True, edge_color=Cycle('Set1'))
        plot = mpl_renderer.get_plot(self.trimesh_weighted.opts(**opts))
        edges = plot.handles['edges']
        colors = np.array([[0.894118, 0.101961, 0.109804, 1.0], [0.215686, 0.494118, 0.721569, 1.0]])
        self.assertEqual(edges.get_facecolors(), colors)

    def test_trimesh_op_node_color(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 'red'), (0, 0, 1, 'green'), (0, 1, 2, 'blue'), (1, 0, 3, 'black')]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='color'))).opts(node_color='color')
        plot = mpl_renderer.get_plot(trimesh)
        artist = plot.handles['nodes']
        self.assertEqual(artist.get_facecolors(), np.array([[1, 0, 0, 1], [0, 0.501961, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]]))

    def test_trimesh_op_node_color_linear(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 2), (0, 0, 1, 1), (0, 1, 2, 3), (1, 0, 3, 4)]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='color'))).opts(node_color='color')
        plot = mpl_renderer.get_plot(trimesh)
        artist = plot.handles['nodes']
        self.assertEqual(np.asarray(artist.get_array()), np.array([2, 1, 3, 4]))
        self.assertEqual(artist.get_clim(), (1, 4))

    def test_trimesh_op_node_color_categorical(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 'B'), (0, 0, 1, 'C'), (0, 1, 2, 'A'), (1, 0, 3, 'B')]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='color'))).opts(node_color='color')
        plot = mpl_renderer.get_plot(trimesh)
        artist = plot.handles['nodes']
        self.assertEqual(np.asarray(artist.get_array()), np.array([0, 1, 2, 0]))
        self.assertEqual(artist.get_clim(), (0, 2))

    def test_trimesh_op_node_size(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 3), (0, 0, 1, 2), (0, 1, 2, 8), (1, 0, 3, 4)]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='size'))).opts(node_size='size')
        plot = mpl_renderer.get_plot(trimesh)
        artist = plot.handles['nodes']
        self.assertEqual(artist.get_sizes(), np.array([9, 4, 64, 16]))

    def test_trimesh_op_node_alpha(self):
        import matplotlib as mpl
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 0.2), (0, 0, 1, 0.6), (0, 1, 2, 1), (1, 0, 3, 0.3)]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='alpha'))).opts(node_alpha='alpha')
        if Version(mpl.__version__) < Version('3.4.0'):
            msg = 'TypeError: alpha must be a float or None'
            with pytest.raises(AbbreviatedException, match=msg):
                mpl_renderer.get_plot(trimesh)
        else:
            plot = mpl_renderer.get_plot(trimesh)
            artist = plot.handles['nodes']
            self.assertEqual(artist.get_alpha(), np.array([0.2, 0.6, 1, 0.3]))

    def test_trimesh_op_node_line_width(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 0.2), (0, 0, 1, 0.6), (0, 1, 2, 1), (1, 0, 3, 0.3)]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='line_width'))).opts(node_linewidth='line_width')
        plot = mpl_renderer.get_plot(trimesh)
        artist = plot.handles['nodes']
        self.assertEqual(artist.get_linewidths(), [0.2, 0.6, 1, 0.3])

    def test_trimesh_op_edge_color_linear_mean_node(self):
        edges = [(0, 1, 2), (1, 2, 3)]
        nodes = [(-1, -1, 0, 2), (0, 0, 1, 1), (0, 1, 2, 3), (1, 0, 3, 4)]
        trimesh = TriMesh((edges, Nodes(nodes, vdims='color'))).opts(edge_color='color')
        plot = mpl_renderer.get_plot(trimesh)
        artist = plot.handles['edges']
        self.assertEqual(np.asarray(artist.get_array()), np.array([2, 8 / 3.0]))
        self.assertEqual(artist.get_clim(), (1, 4))

    def test_trimesh_op_edge_color(self):
        edges = [(0, 1, 2, 'red'), (1, 2, 3, 'blue')]
        nodes = [(-1, -1, 0), (0, 0, 1), (0, 1, 2), (1, 0, 3)]
        trimesh = TriMesh((edges, nodes), vdims='color').opts(edge_color='color')
        plot = mpl_renderer.get_plot(trimesh)
        artist = plot.handles['edges']
        self.assertEqual(artist.get_edgecolors(), np.array([[1, 0, 0, 1], [0, 0, 1, 1]]))

    def test_trimesh_op_edge_color_linear(self):
        edges = [(0, 1, 2, 2.4), (1, 2, 3, 3.6)]
        nodes = [(-1, -1, 0), (0, 0, 1), (0, 1, 2), (1, 0, 3)]
        trimesh = TriMesh((edges, nodes), vdims='color').opts(edge_color='color')
        plot = mpl_renderer.get_plot(trimesh)
        artist = plot.handles['edges']
        self.assertEqual(np.asarray(artist.get_array()), np.array([2.4, 3.6]))
        self.assertEqual(artist.get_clim(), (2.4, 3.6))

    def test_trimesh_op_edge_color_categorical(self):
        edges = [(0, 1, 2, 'A'), (1, 2, 3, 'B')]
        nodes = [(-1, -1, 0), (0, 0, 1), (0, 1, 2), (1, 0, 3)]
        trimesh = TriMesh((edges, nodes), vdims='color').opts(edge_color='color')
        plot = mpl_renderer.get_plot(trimesh)
        artist = plot.handles['edges']
        self.assertEqual(np.asarray(artist.get_array()), np.array([0, 1]))
        self.assertEqual(artist.get_clim(), (0, 1))

    def test_trimesh_op_edge_alpha(self):
        edges = [(0, 1, 2, 0.7), (1, 2, 3, 0.3)]
        nodes = [(-1, -1, 0), (0, 0, 1), (0, 1, 2), (1, 0, 3)]
        trimesh = TriMesh((edges, nodes), vdims='alpha').opts(edge_alpha='alpha')
        msg = 'ValueError: Mapping a dimension to the "edge_alpha" style'
        with pytest.raises(AbbreviatedException, match=msg):
            mpl_renderer.get_plot(trimesh)

    def test_trimesh_op_edge_line_width(self):
        edges = [(0, 1, 2, 7), (1, 2, 3, 3)]
        nodes = [(-1, -1, 0), (0, 0, 1), (0, 1, 2), (1, 0, 3)]
        trimesh = TriMesh((edges, nodes), vdims='line_width').opts(edge_linewidth='line_width')
        plot = mpl_renderer.get_plot(trimesh)
        artist = plot.handles['edges']
        self.assertEqual(artist.get_linewidths(), [7, 3])