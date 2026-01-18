import pathlib
import sys
from unittest import TestCase, SkipTest
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
from hvplot.util import proj_to_cartopy
from packaging.version import Version
class TestGeoAnnotation(TestCase):

    def setUp(self):
        try:
            import geoviews
            import cartopy.crs as ccrs
        except:
            raise SkipTest('geoviews or cartopy not available')
        import hvplot.pandas
        self.crs = ccrs.PlateCarree()
        self.df = pd.DataFrame(np.random.rand(10, 2), columns=['x', 'y'])

    def test_plot_with_coastline(self):
        import geoviews as gv
        plot = self.df.hvplot.points('x', 'y', geo=True, coastline=True)
        self.assertEqual(len(plot), 2)
        coastline = plot.get(1)
        self.assertIsInstance(coastline, gv.Feature)

    def test_plot_with_coastline_sets_geo_by_default(self):
        import geoviews as gv
        plot = self.df.hvplot.points('x', 'y', coastline=True)
        self.assertEqual(len(plot), 2)
        coastline = plot.get(1)
        self.assertIsInstance(coastline, gv.Feature)

    def test_plot_with_coastline_scale(self):
        plot = self.df.hvplot.points('x', 'y', geo=True, coastline='10m')
        opts = plot.get(1).opts.get('plot')
        assert opts.kwargs['scale'] == '10m'

    def test_plot_with_tiles(self):
        plot = self.df.hvplot.points('x', 'y', geo=False, tiles=True)
        self.assertEqual(len(plot), 2)
        self.assertIsInstance(plot.get(0), hv.Tiles)
        self.assertIn('openstreetmap', plot.get(0).data)

    def test_plot_with_tiles_with_geo(self):
        import geoviews as gv
        plot = self.df.hvplot.points('x', 'y', geo=True, tiles=True)
        self.assertEqual(len(plot), 2)
        self.assertIsInstance(plot.get(0), gv.element.WMTS)
        self.assertIn('openstreetmap', plot.get(0).data)

    def test_plot_with_specific_tiles(self):
        plot = self.df.hvplot.points('x', 'y', geo=False, tiles='ESRI')
        self.assertEqual(len(plot), 2)
        self.assertIsInstance(plot.get(0), hv.Tiles)
        self.assertIn('ArcGIS', plot.get(0).data)

    def test_plot_with_specific_tiles_geo(self):
        import geoviews as gv
        plot = self.df.hvplot.points('x', 'y', geo=True, tiles='ESRI')
        self.assertEqual(len(plot), 2)
        self.assertIsInstance(plot.get(0), gv.element.WMTS)
        self.assertIn('ArcGIS', plot.get(0).data)

    def test_plot_with_specific_tile_class(self):
        plot = self.df.hvplot.points('x', 'y', geo=False, tiles=hv.element.tiles.EsriImagery)
        self.assertEqual(len(plot), 2)
        self.assertIsInstance(plot.get(0), hv.Tiles)
        self.assertIn('ArcGIS', plot.get(0).data)

    def test_plot_with_specific_tile_class_with_geo(self):
        import geoviews as gv
        plot = self.df.hvplot.points('x', 'y', geo=True, tiles=gv.tile_sources.EsriImagery)
        self.assertEqual(len(plot), 2)
        self.assertIsInstance(plot.get(0), gv.element.WMTS)
        self.assertIn('ArcGIS', plot.get(0).data)

    def test_plot_with_specific_tile_obj(self):
        plot = self.df.hvplot.points('x', 'y', geo=False, tiles=hv.element.tiles.EsriImagery())
        self.assertEqual(len(plot), 2)
        self.assertIsInstance(plot.get(0), hv.Tiles)
        self.assertIn('ArcGIS', plot.get(0).data)

    def test_plot_with_specific_tile_obj_with_geo(self):
        plot = self.df.hvplot.points('x', 'y', geo=True, tiles=hv.element.tiles.EsriImagery())
        self.assertEqual(len(plot), 2)
        self.assertIsInstance(plot.get(0), hv.Tiles)
        self.assertIn('ArcGIS', plot.get(0).data)

    def test_plot_with_specific_gv_tile_obj(self):
        import geoviews as gv
        plot = self.df.hvplot.points('x', 'y', geo=True, tiles=gv.tile_sources.CartoDark)
        self.assertEqual(len(plot), 2)
        self.assertIsInstance(plot.get(0), gv.element.WMTS)

    def test_plot_with_features_properly_overlaid_underlaid(self):
        plot = self.df.hvplot.points('x', 'y', features=['land', 'borders'])
        assert plot.get(0).group == 'Land'
        assert plot.get(2).group == 'Borders'