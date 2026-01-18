import numpy as np
from holoviews.element import (
from .test_plot import TestPlotlyPlot
class TestMapboxPathShape(TestMapboxShape):

    def test_simple_path(self):
        path = Tiles('') * Path([(self.x_range[0], self.y_range[0]), (self.x_range[1], self.y_range[1]), (self.x_range[0], self.y_range[1]), (self.x_range[0], self.y_range[0])]).redim.range(x=self.x_range, y=self.y_range)
        state = self._get_plot_state(path)
        self.assertEqual(state['data'][1]['type'], 'scattermapbox')
        self.assertEqual(state['data'][1]['mode'], 'lines')
        self.assertEqual(state['data'][1]['lon'], np.array([self.lon_range[i] for i in (0, 1, 0, 0)] + [np.nan]))
        self.assertEqual(state['data'][1]['lat'], np.array([self.lat_range[i] for i in (0, 1, 1, 0)] + [np.nan]))
        self.assertEqual(state['data'][1]['showlegend'], False)
        self.assertEqual(state['data'][1]['line']['color'], default_shape_color)
        self.assertEqual(state['layout']['mapbox']['center'], {'lat': self.lat_center, 'lon': self.lon_center})