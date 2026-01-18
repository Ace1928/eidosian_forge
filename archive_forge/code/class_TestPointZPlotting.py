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
class TestPointZPlotting:

    def setup_method(self):
        self.N = 10
        self.points = GeoSeries((Point(i, i, i) for i in range(self.N)))
        values = np.arange(self.N)
        self.df = GeoDataFrame({'geometry': self.points, 'values': values})

    def test_plot(self):
        self.df.plot()