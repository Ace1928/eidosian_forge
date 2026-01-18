import numpy as np
from holoviews.element import Scatter, Tiles
from .test_plot import TestPlotlyPlot
class TestMapboxScatterPlot(TestPlotlyPlot):

    def test_scatter_state(self):
        xs = [3000000, 2000000, 1000000]
        ys = [-3000000, -2000000, -1000000]
        x_range = (-5000000, 4000000)
        x_center = sum(x_range) / 2.0
        y_range = (-3000000, 2000000)
        y_center = sum(y_range) / 2.0
        lon_centers, lat_centers = Tiles.easting_northing_to_lon_lat([x_center], [y_center])
        lon_center, lat_center = (lon_centers[0], lat_centers[0])
        lons, lats = Tiles.easting_northing_to_lon_lat(xs, ys)
        scatter = Tiles('') * Scatter((xs, ys)).redim.range(x=x_range, y=y_range)
        state = self._get_plot_state(scatter)
        self.assertEqual(state['data'][1]['type'], 'scattermapbox')
        self.assertEqual(state['data'][1]['lon'], lons)
        self.assertEqual(state['data'][1]['lat'], lats)
        self.assertEqual(state['data'][1]['mode'], 'markers')
        self.assertEqual(state['layout']['mapbox']['center'], {'lat': lat_center, 'lon': lon_center})
        self.assertFalse('xaxis' in state['layout'])
        self.assertFalse('yaxis' in state['layout'])

    def test_scatter_color_mapped(self):
        scatter = Tiles('') * Scatter([3, 2, 1]).opts(color='x')
        state = self._get_plot_state(scatter)
        self.assertEqual(state['data'][1]['marker']['color'], np.array([0, 1, 2]))
        self.assertEqual(state['data'][1]['marker']['cmin'], 0)
        self.assertEqual(state['data'][1]['marker']['cmax'], 2)

    def test_scatter_size(self):
        scatter = Tiles('') * Scatter([3, 2, 1]).opts(size='y')
        state = self._get_plot_state(scatter)
        self.assertEqual(state['data'][1]['marker']['size'], np.array([3, 2, 1]))

    def test_scatter_colors(self):
        scatter = Tiles('') * Scatter([(0, 1, 'red'), (1, 2, 'green'), (2, 3, 'blue')], vdims=['y', 'color']).opts(color='color')
        state = self._get_plot_state(scatter)
        self.assertEqual(np.array_equal(state['data'][1]['marker']['color'], np.array(['red', 'green', 'blue'])), True)

    def test_scatter_markers(self):
        scatter = Tiles('') * Scatter([(0, 1, 'square'), (1, 2, 'circle'), (2, 3, 'triangle-up')], vdims=['y', 'marker']).opts(marker='marker')
        state = self._get_plot_state(scatter)
        self.assertEqual(np.array_equal(state['data'][1]['marker']['symbol'], np.array(['square', 'circle', 'triangle-up'])), True)

    def test_scatter_selectedpoints(self):
        scatter = Tiles('') * Scatter([(0, 1), (1, 2), (2, 3)]).opts(selectedpoints=[1, 2])
        state = self._get_plot_state(scatter)
        self.assertEqual(state['data'][1]['selectedpoints'], [1, 2])

    def test_visible(self):
        element = Tiles('') * Scatter([3, 2, 1]).opts(visible=False)
        state = self._get_plot_state(element)
        self.assertEqual(state['data'][1]['visible'], False)