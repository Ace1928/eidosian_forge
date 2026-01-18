import numpy as np
import PIL.Image
import plotly.graph_objs as go
from holoviews.element import RGB, Tiles
from .test_plot import TestPlotlyPlot, plotly_renderer
class TestMapboxRGBPlot(TestPlotlyPlot):

    def setUp(self):
        super().setUp()
        self.xs = [3000000, 2000000, 1000000]
        self.ys = [-3000000, -2000000, -1000000]
        self.x_range = (-5000000, 4000000)
        self.x_center = sum(self.x_range) / 2.0
        self.y_range = (-3000000, 2000000)
        self.y_center = sum(self.y_range) / 2.0
        self.lon_range, self.lat_range = Tiles.easting_northing_to_lon_lat(self.x_range, self.y_range)
        self.lon_centers, self.lat_centers = Tiles.easting_northing_to_lon_lat([self.x_center], [self.y_center])
        self.lon_center, self.lat_center = (self.lon_centers[0], self.lat_centers[0])
        self.lons, self.lats = Tiles.easting_northing_to_lon_lat(self.xs, self.ys)

    def test_rgb(self):
        rgb_data = np.random.rand(10, 10, 3)
        rgb = Tiles('') * RGB(rgb_data, bounds=(self.x_range[0], self.y_range[0], self.x_range[1], self.y_range[1])).opts(opacity=0.5).redim.range(x=self.x_range, y=self.y_range)
        fig_dict = plotly_renderer.get_plot_state(rgb)
        self.assertEqual(fig_dict['data'][1]['type'], 'scattermapbox')
        self.assertEqual(fig_dict['data'][1]['lon'], [None])
        self.assertEqual(fig_dict['data'][1]['lat'], [None])
        self.assertEqual(fig_dict['data'][1]['showlegend'], False)
        subplot = fig_dict['layout']['mapbox']
        self.assertEqual(subplot['style'], 'white-bg')
        self.assertEqual(subplot['center'], {'lat': self.lat_center, 'lon': self.lon_center})
        layers = fig_dict['layout']['mapbox']['layers']
        self.assertEqual(len(layers), 1)
        rgb_layer = layers[0]
        self.assertEqual(rgb_layer['below'], 'traces')
        self.assertEqual(rgb_layer['coordinates'], [[self.lon_range[0], self.lat_range[1]], [self.lon_range[1], self.lat_range[1]], [self.lon_range[1], self.lat_range[0]], [self.lon_range[0], self.lat_range[0]]])
        self.assertTrue(rgb_layer['source'].startswith('data:image/png;base64,iVBOR'))
        self.assertEqual(rgb_layer['opacity'], 0.5)
        self.assertEqual(rgb_layer['sourcetype'], 'image')