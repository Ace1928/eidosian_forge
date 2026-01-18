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
class TestPathPlot(TestBokehPlot):

    def test_batched_path_line_color_and_color(self):
        opts = {'NdOverlay': dict(legend_limit=0), 'Path': dict(line_color=Cycle(values=['red', 'blue']))}
        overlay = NdOverlay({i: Path([[(i, j) for j in range(2)]]) for i in range(2)}).opts(opts)
        plot = bokeh_renderer.get_plot(overlay).subplots[()]
        line_color = ['red', 'blue']
        self.assertEqual(plot.handles['source'].data['line_color'], line_color)

    def test_batched_path_alpha_and_color(self):
        opts = {'NdOverlay': dict(legend_limit=0), 'Path': dict(alpha=Cycle(values=[0.5, 1]))}
        overlay = NdOverlay({i: Path([[(i, j) for j in range(2)]]) for i in range(2)}).opts(opts)
        plot = bokeh_renderer.get_plot(overlay).subplots[()]
        alpha = [0.5, 1.0]
        color = ['#30a2da', '#fc4f30']
        self.assertEqual(plot.handles['source'].data['alpha'], alpha)
        self.assertEqual(plot.handles['source'].data['color'], color)

    def test_batched_path_line_width_and_color(self):
        opts = {'NdOverlay': dict(legend_limit=0), 'Path': dict(line_width=Cycle(values=[0.5, 1]))}
        overlay = NdOverlay({i: Path([[(i, j) for j in range(2)]]) for i in range(2)}).opts(opts)
        plot = bokeh_renderer.get_plot(overlay).subplots[()]
        line_width = [0.5, 1.0]
        color = ['#30a2da', '#fc4f30']
        self.assertEqual(plot.handles['source'].data['line_width'], line_width)
        self.assertEqual(plot.handles['source'].data['color'], color)

    def test_path_overlay_hover(self):
        obj = NdOverlay({i: Path([np.random.rand(10, 2)]) for i in range(5)}, kdims=['Test'])
        opts = {'Path': {'tools': ['hover']}, 'NdOverlay': {'legend_limit': 0}}
        obj = obj.opts(opts)
        self._test_hover_info(obj, [('Test', '@{Test}')])

    def test_path_colored_and_split_with_extra_vdims(self):
        xs = [1, 2, 3, 4]
        ys = xs[::-1]
        color = [0, 0.25, 0.5, 0.75]
        other = ['A', 'B', 'C', 'D']
        data = {'x': xs, 'y': ys, 'color': color, 'other': other}
        path = Path([data], vdims=['color', 'other']).opts(color_index='color', tools=['hover'])
        plot = bokeh_renderer.get_plot(path)
        source = plot.handles['source']
        self.assertEqual(source.data['xs'], [np.array([1, 2]), np.array([2, 3]), np.array([3, 4])])
        self.assertEqual(source.data['ys'], [np.array([4, 3]), np.array([3, 2]), np.array([2, 1])])
        self.assertEqual(source.data['other'], np.array(['A', 'B', 'C']))
        self.assertEqual(source.data['color'], np.array([0, 0.25, 0.5]))

    def test_path_colored_dim_split_with_extra_vdims(self):
        xs = [1, 2, 3, 4]
        ys = xs[::-1]
        color = [0, 0.25, 0.5, 0.75]
        other = ['A', 'B', 'C', 'D']
        data = {'x': xs, 'y': ys, 'color': color, 'other': other}
        path = Path([data], vdims=['color', 'other']).opts(color=dim('color') * 2, tools=['hover'])
        plot = bokeh_renderer.get_plot(path)
        source = plot.handles['source']
        self.assertEqual(source.data['xs'], [np.array([1, 2]), np.array([2, 3]), np.array([3, 4])])
        self.assertEqual(source.data['ys'], [np.array([4, 3]), np.array([3, 2]), np.array([2, 1])])
        self.assertEqual(source.data['other'], np.array(['A', 'B', 'C']))
        self.assertEqual(source.data['color'], np.array([0, 0.5, 1]))

    def test_path_colored_by_levels_single_value(self):
        xs = [1, 2, 3, 4]
        ys = xs[::-1]
        color = [998, 999, 998, 998]
        date = np.datetime64(dt.datetime(2018, 8, 1))
        data = {'x': xs, 'y': ys, 'color': color, 'date': date}
        levels = [0, 38, 73, 95, 110, 130, 156, 999]
        colors = ['#5ebaff', '#00faf4', '#ffffcc', '#ffe775', '#ffc140', '#ff8f20', '#ff6060']
        path = Path([data], vdims=['color', 'date']).opts(color_index='color', color_levels=levels, cmap=colors, tools=['hover'])
        plot = bokeh_renderer.get_plot(path)
        source = plot.handles['source']
        cmapper = plot.handles['color_mapper']
        self.assertEqual(source.data['xs'], [np.array([1, 2]), np.array([2, 3]), np.array([3, 4])])
        self.assertEqual(source.data['ys'], [np.array([4, 3]), np.array([3, 2]), np.array([2, 1])])
        self.assertEqual(source.data['color'], np.array([998, 999, 998]))
        self.assertEqual(source.data['date'], np.array([date] * 3))
        self.assertEqual(cmapper.low, 998)
        self.assertEqual(cmapper.high, 999)
        self.assertEqual(cmapper.palette, colors[-1:])

    def test_path_continuously_varying_color_op(self):
        xs = [1, 2, 3, 4]
        ys = xs[::-1]
        color = [998, 999, 998, 994]
        date = np.datetime64(dt.datetime(2018, 8, 1))
        data = {'x': xs, 'y': ys, 'color': color, 'date': date}
        levels = [0, 38, 73, 95, 110, 130, 156, 999]
        colors = ['#5ebaff', '#00faf4', '#ffffcc', '#ffe775', '#ffc140', '#ff8f20', '#ff6060']
        path = Path([data], vdims=['color', 'date']).opts(color='color', color_levels=levels, cmap=colors, tools=['hover'])
        plot = bokeh_renderer.get_plot(path)
        source = plot.handles['source']
        cmapper = plot.handles['color_color_mapper']
        self.assertEqual(source.data['xs'], [np.array([1, 2]), np.array([2, 3]), np.array([3, 4])])
        self.assertEqual(source.data['ys'], [np.array([4, 3]), np.array([3, 2]), np.array([2, 1])])
        self.assertEqual(source.data['color'], np.array([998, 999, 998]))
        self.assertEqual(source.data['date'], np.array([date] * 3))
        self.assertEqual(cmapper.low, 994)
        self.assertEqual(cmapper.high, 999)
        self.assertEqual(cmapper.palette, colors[-1:])

    def test_path_continuously_varying_alpha_op(self):
        xs = [1, 2, 3, 4]
        ys = xs[::-1]
        alpha = [0.1, 0.7, 0.3, 0.2]
        data = {'x': xs, 'y': ys, 'alpha': alpha}
        path = Path([data], vdims='alpha').opts(alpha='alpha')
        plot = bokeh_renderer.get_plot(path)
        source = plot.handles['source']
        self.assertEqual(source.data['xs'], [np.array([1, 2]), np.array([2, 3]), np.array([3, 4])])
        self.assertEqual(source.data['ys'], [np.array([4, 3]), np.array([3, 2]), np.array([2, 1])])
        self.assertEqual(source.data['alpha'], np.array([0.1, 0.7, 0.3]))

    def test_path_continuously_varying_line_width_op(self):
        xs = [1, 2, 3, 4]
        ys = xs[::-1]
        line_width = [1, 7, 3, 2]
        data = {'x': xs, 'y': ys, 'line_width': line_width}
        path = Path([data], vdims='line_width').opts(line_width='line_width')
        plot = bokeh_renderer.get_plot(path)
        source = plot.handles['source']
        self.assertEqual(source.data['xs'], [np.array([1, 2]), np.array([2, 3]), np.array([3, 4])])
        self.assertEqual(source.data['ys'], [np.array([4, 3]), np.array([3, 2]), np.array([2, 1])])
        self.assertEqual(source.data['line_width'], np.array([1, 7, 3]))

    def test_path_continuously_varying_color_legend(self):
        data = {'x': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'y': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'cat': [0, 1, 2, 0, 1, 2, 0, 1, 2]}
        colors = ['#FF0000', '#00FF00', '#0000FF']
        levels = [0, 1, 2, 3]
        path = Path(data, vdims='cat').opts(color='cat', cmap=dict(zip(levels, colors)), line_width=4, show_legend=True)
        plot = bokeh_renderer.get_plot(path)
        item = plot.state.legend[0].items[0]
        legend = {'field': 'color_str__'}
        self.assertEqual(property_to_dict(item.label), legend)
        self.assertEqual(item.renderers, [plot.handles['glyph_renderer']])

    def test_path_continuously_varying_color_legend_with_labels(self):
        data = {'x': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'y': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'cat': [0, 1, 2, 0, 1, 2, 0, 1, 2]}
        colors = ['#FF0000', '#00FF00', '#0000FF']
        levels = [0, 1, 2, 3]
        path = Path(data, vdims='cat').opts(color='cat', cmap=dict(zip(levels, colors)), line_width=4, show_legend=True, legend_labels={0: 'A', 1: 'B', 2: 'C'})
        plot = bokeh_renderer.get_plot(path)
        cds = plot.handles['cds']
        item = plot.state.legend[0].items[0]
        legend = {'field': '_color_str___labels'}
        self.assertEqual(cds.data['_color_str___labels'], ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'])
        self.assertEqual(property_to_dict(item.label), legend)
        self.assertEqual(item.renderers, [plot.handles['glyph_renderer']])