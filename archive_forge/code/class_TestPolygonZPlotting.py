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
class TestPolygonZPlotting:

    def setup_method(self):
        t1 = Polygon([(0, 0, 0), (1, 0, 0), (1, 1, 1)])
        t2 = Polygon([(1, 0, 0), (2, 0, 0), (2, 1, 1)])
        self.polys = GeoSeries([t1, t2], index=list('AB'))
        self.df = GeoDataFrame({'geometry': self.polys, 'values': [0, 1]})
        multipoly1 = MultiPolygon([t1, t2])
        multipoly2 = rotate(multipoly1, 180)
        self.df2 = GeoDataFrame({'geometry': [multipoly1, multipoly2], 'values': [0, 1]})

    def test_plot(self):
        self.df.plot()