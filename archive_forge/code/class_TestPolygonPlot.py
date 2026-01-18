import datetime as dt
import numpy as np
import pandas as pd
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from holoviews.core import HoloMap, NdOverlay
from holoviews.core.options import Cycle
from holoviews.element import Contours, Path, Polygons
from holoviews.plotting.bokeh.util import property_to_dict
from holoviews.streams import PolyDraw
from holoviews.util.transform import dim
from .test_plot import TestBokehPlot, bokeh_renderer
class TestPolygonPlot(TestBokehPlot):

    def test_polygons_overlay_hover(self):
        obj = NdOverlay({i: Polygons([{('x', 'y'): np.random.rand(10, 2), 'z': 0}], vdims=['z']) for i in range(5)}, kdims=['Test'])
        opts = {'Polygons': {'tools': ['hover']}, 'NdOverlay': {'legend_limit': 0}}
        obj = obj.opts(opts)
        self._test_hover_info(obj, [('Test', '@{Test}'), ('z', '@{z}')])

    def test_polygons_colored(self):
        polygons = NdOverlay({j: Polygons([[(i ** j, i, j) for i in range(10)]], vdims='Value') for j in range(5)})
        plot = bokeh_renderer.get_plot(polygons)
        for i, splot in enumerate(plot.subplots.values()):
            cmapper = splot.handles['color_mapper']
            self.assertEqual(cmapper.low, 0)
            self.assertEqual(cmapper.high, 4)
            source = splot.handles['source']
            self.assertEqual(source.data['Value'], np.array([i]))

    def test_polygons_colored_batched(self):
        polygons = NdOverlay({j: Polygons([[(i ** j, i, j) for i in range(10)]], vdims='Value') for j in range(5)}).opts(legend_limit=0)
        plot = next(iter(bokeh_renderer.get_plot(polygons).subplots.values()))
        cmapper = plot.handles['color_mapper']
        self.assertEqual(cmapper.low, 0)
        self.assertEqual(cmapper.high, 4)
        source = plot.handles['source']
        self.assertEqual(plot.handles['glyph'].fill_color['transform'], cmapper)
        self.assertEqual(source.data['Value'], list(range(5)))

    def test_polygons_colored_batched_unsanitized(self):
        polygons = NdOverlay({j: Polygons([[(i ** j, i, j) for i in range(10)] for i in range(2)], vdims=['some ? unescaped name']) for j in range(5)}).opts(legend_limit=0)
        plot = next(iter(bokeh_renderer.get_plot(polygons).subplots.values()))
        cmapper = plot.handles['color_mapper']
        self.assertEqual(cmapper.low, 0)
        self.assertEqual(cmapper.high, 4)
        source = plot.handles['source']
        self.assertEqual(source.data['some_question_mark_unescaped_name'], [j for i in range(5) for j in [i, i]])

    def test_empty_polygons_plot(self):
        poly = Polygons([], vdims=['Intensity'])
        plot = bokeh_renderer.get_plot(poly)
        source = plot.handles['source']
        self.assertEqual(len(source.data['xs']), 0)
        self.assertEqual(len(source.data['ys']), 0)
        self.assertEqual(len(source.data['Intensity']), 0)

    def test_polygon_with_hole_plot(self):
        xs = [1, 2, 3]
        ys = [2, 0, 7]
        holes = [[[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]]]
        poly = Polygons([{'x': xs, 'y': ys, 'holes': holes}])
        plot = bokeh_renderer.get_plot(poly)
        source = plot.handles['source']
        self.assertEqual(source.data['xs'], [[[np.array([1, 2, 3, 1]), np.array([1.5, 2, 1.6, 1.5]), np.array([2.1, 2.5, 2.3, 2.1])]]])
        self.assertEqual(source.data['ys'], [[[np.array([2, 0, 7, 2]), np.array([2, 3, 1.6, 2]), np.array([4.5, 5, 3.5, 4.5])]]])

    def test_multi_polygon_hole_plot(self):
        xs = [1, 2, 3, np.nan, 3, 7, 6]
        ys = [2, 0, 7, np.nan, 2, 5, 7]
        holes = [[[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]], []]
        poly = Polygons([{'x': xs, 'y': ys, 'holes': holes}])
        plot = bokeh_renderer.get_plot(poly)
        source = plot.handles['source']
        self.assertEqual(source.data['xs'], [[[np.array([1, 2, 3, 1]), np.array([1.5, 2, 1.6, 1.5]), np.array([2.1, 2.5, 2.3, 2.1])], [np.array([3, 7, 6, 3])]]])
        self.assertEqual(source.data['ys'], [[[np.array([2, 0, 7, 2]), np.array([2, 3, 1.6, 2]), np.array([4.5, 5, 3.5, 4.5])], [np.array([2, 5, 7, 2])]]])

    def test_polygons_hover_color_op(self):
        polygons = Polygons([{('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 'green'}, {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 'red'}], vdims='color').opts(fill_color='color', tools=['hover'])
        plot = bokeh_renderer.get_plot(polygons)
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.line_color, 'black')
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'fill_color'})
        self.assertEqual(cds.data['color'], np.array(['green', 'red']))
        self.assertEqual(cds.data['fill_color'], np.array(['green', 'red']))

    def test_polygons_color_op(self):
        polygons = Polygons([{('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 'green'}, {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 'red'}], vdims='color').opts(color='color')
        plot = bokeh_renderer.get_plot(polygons)
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        self.assertEqual(glyph.line_color, 'black')
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'color'})
        self.assertEqual(cds.data['color'], np.array(['green', 'red']))

    def test_polygons_linear_color_op(self):
        polygons = Polygons([{('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 7}, {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 3}], vdims='color').opts(color='color')
        plot = bokeh_renderer.get_plot(polygons)
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_color_mapper']
        self.assertEqual(glyph.line_color, 'black')
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'color', 'transform': cmapper})
        self.assertEqual(cds.data['color'], np.array([7, 3]))
        self.assertIsInstance(cmapper, LinearColorMapper)
        self.assertEqual(cmapper.low, 3)
        self.assertEqual(cmapper.high, 7)

    def test_polygons_categorical_color_op(self):
        polygons = Polygons([{('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'color': 'b'}, {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'color': 'a'}], vdims='color').opts(color='color')
        plot = bokeh_renderer.get_plot(polygons)
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        cmapper = plot.handles['color_color_mapper']
        self.assertEqual(glyph.line_color, 'black')
        self.assertEqual(property_to_dict(glyph.fill_color), {'field': 'color', 'transform': cmapper})
        self.assertEqual(cds.data['color'], np.array(['b', 'a']))
        self.assertIsInstance(cmapper, CategoricalColorMapper)
        self.assertEqual(cmapper.factors, ['b', 'a'])

    def test_polygons_alpha_op(self):
        polygons = Polygons([{('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'alpha': 0.7}, {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'alpha': 0.3}], vdims='alpha').opts(alpha='alpha')
        plot = bokeh_renderer.get_plot(polygons)
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        self.assertEqual(property_to_dict(glyph.line_alpha), {'field': 'alpha'})
        self.assertEqual(property_to_dict(glyph.fill_alpha), {'field': 'alpha'})
        self.assertEqual(cds.data['alpha'], np.array([0.7, 0.3]))

    def test_polygons_line_width_op(self):
        polygons = Polygons([{('x', 'y'): [(0, 0), (0, 1), (1, 0)], 'line_width': 7}, {('x', 'y'): [(1, 0), (1, 1), (0, 1)], 'line_width': 3}], vdims='line_width').opts(line_width='line_width')
        plot = bokeh_renderer.get_plot(polygons)
        cds = plot.handles['source']
        glyph = plot.handles['glyph']
        self.assertEqual(property_to_dict(glyph.line_width), {'field': 'line_width'})
        self.assertEqual(cds.data['line_width'], np.array([7, 3]))

    def test_polygons_holes_initialize(self):
        from bokeh.models import MultiPolygons
        xs = [1, 2, 3, np.nan, 6, 7, 3]
        ys = [2, 0, 7, np.nan, 7, 5, 2]
        holes = [[[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]], []]
        poly = HoloMap({0: Polygons([{'x': xs, 'y': ys, 'holes': holes}]), 1: Polygons([{'x': xs, 'y': ys}])})
        plot = bokeh_renderer.get_plot(poly)
        glyph = plot.handles['glyph']
        self.assertTrue(plot._has_holes)
        self.assertIsInstance(glyph, MultiPolygons)

    def test_polygons_no_holes_with_draw_tool(self):
        from bokeh.models import Patches
        xs = [1, 2, 3, np.nan, 6, 7, 3]
        ys = [2, 0, 7, np.nan, 7, 5, 2]
        holes = [[[(1.5, 2), (2, 3), (1.6, 1.6)], [(2.1, 4.5), (2.5, 5), (2.3, 3.5)]], []]
        poly = HoloMap({0: Polygons([{'x': xs, 'y': ys, 'holes': holes}]), 1: Polygons([{'x': xs, 'y': ys}])})
        PolyDraw(source=poly)
        plot = bokeh_renderer.get_plot(poly)
        glyph = plot.handles['glyph']
        self.assertFalse(plot._has_holes)
        self.assertIsInstance(glyph, Patches)