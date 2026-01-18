import numpy as np
from holoviews.element import (
from .test_plot import TestPlotlyPlot
class TestMapboxSegments(TestMapboxShape):

    def test_segments_simple(self):
        rectangles = Tiles('') * Segments([(0, 0, self.x_range[1], self.y_range[1]), (self.x_range[0], self.y_range[0], 0, 0)]).redim.range(x=self.x_range, y=self.y_range)
        state = self._get_plot_state(rectangles)
        self.assertEqual(state['data'][1]['type'], 'scattermapbox')
        self.assertEqual(state['data'][1]['mode'], 'lines')
        self.assertEqual(state['data'][1]['showlegend'], False)
        self.assertEqual(state['data'][1]['line']['color'], default_shape_color)
        self.assertEqual(state['data'][1]['lon'], np.array([0, self.lon_range[1], np.nan, self.lon_range[0], 0, np.nan]))
        self.assertEqual(state['data'][1]['lat'], np.array([0, self.lat_range[1], np.nan, self.lat_range[0], 0, np.nan]))
        self.assertEqual(state['layout']['mapbox']['center'], {'lat': self.lat_center, 'lon': self.lon_center})