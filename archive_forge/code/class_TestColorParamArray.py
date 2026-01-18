import itertools
from packaging.version import Version
import warnings
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.affinity import rotate
from shapely.geometry import (
from geopandas import GeoDataFrame, GeoSeries, read_file
from geopandas.datasets import get_path
import geopandas._compat as compat
from geopandas.plotting import GeoplotAccessor
import pytest
import matplotlib.pyplot as plt
class TestColorParamArray:

    def setup_method(self):
        geom = []
        color = []
        for a, b in [(0, 2), (4, 6)]:
            b = box(a, a, b, b)
            geom += [b, b.buffer(0.8).exterior, b.centroid]
            color += ['red', 'green', 'blue']
        self.gdf = GeoDataFrame({'geometry': geom, 'color_rgba': color})
        self.mgdf = self.gdf.dissolve(self.gdf.geom_type)

    def test_color_single(self):
        ax = self.gdf.plot(color=self.gdf['color_rgba'])
        _check_colors(4, np.concatenate([c.get_edgecolor() for c in ax.collections]), ['green'] * 2 + ['blue'] * 2)
        _check_colors(4, np.concatenate([c.get_facecolor() for c in ax.collections]), ['red'] * 2 + ['blue'] * 2)

    def test_color_multi(self):
        ax = self.mgdf.plot(color=self.mgdf['color_rgba'])
        _check_colors(4, np.concatenate([c.get_edgecolor() for c in ax.collections]), ['green'] * 2 + ['blue'] * 2)
        _check_colors(4, np.concatenate([c.get_facecolor() for c in ax.collections]), ['red'] * 2 + ['blue'] * 2)