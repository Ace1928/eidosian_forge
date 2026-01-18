import numpy as np
from holoviews.element import (
from .test_plot import TestPlotlyPlot
class TestMapboxBounds(TestMapboxShape):

    def test_single_bounds(self):
        bounds = Tiles('') * Bounds((self.x_range[0], self.y_range[0], self.x_range[1], self.y_range[1])).redim.range(x=self.x_range, y=self.y_range)
        state = self._get_plot_state(bounds)
        self.assertEqual(state['data'][1]['type'], 'scattermapbox')
        self.assertEqual(state['data'][1]['mode'], 'lines')
        self.assertEqual(state['data'][1]['lon'], np.array([self.lon_range[i] for i in (0, 0, 1, 1, 0)]))
        self.assertEqual(state['data'][1]['lat'], np.array([self.lat_range[i] for i in (0, 1, 1, 0, 0)]))
        self.assertEqual(state['data'][1]['showlegend'], False)
        self.assertEqual(state['data'][1]['line']['color'], default_shape_color)
        self.assertEqual(state['layout']['mapbox']['center'], {'lat': self.lat_center, 'lon': self.lon_center})

    def test_bounds_layout(self):
        bounds1 = Bounds((0, 0, 1, 1))
        bounds2 = Bounds((0, 0, 2, 2))
        bounds3 = Bounds((0, 0, 3, 3))
        bounds4 = Bounds((0, 0, 4, 4))
        layout = (Tiles('') * bounds1 + Tiles('') * bounds2 + Tiles('') * bounds3 + Tiles('') * bounds4).cols(2)
        state = self._get_plot_state(layout)
        self.assertEqual(state['data'][1]['subplot'], 'mapbox')
        self.assertEqual(state['data'][3]['subplot'], 'mapbox2')
        self.assertEqual(state['data'][5]['subplot'], 'mapbox3')
        self.assertEqual(state['data'][7]['subplot'], 'mapbox4')
        self.assertNotIn('xaxis', state['layout'])
        self.assertNotIn('yaxis', state['layout'])