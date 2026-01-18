import numpy as np
from holoviews.element import (
from .test_plot import TestPlotlyPlot
class TestMapboxShape(TestPlotlyPlot):

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