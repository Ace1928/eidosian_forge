import datetime
import io
import os
import pathlib
import tempfile
from collections import OrderedDict
import numpy as np
import pandas as pd
import pytest
import pytz
from packaging.version import Version
from pandas.api.types import is_datetime64_any_dtype
from pandas.testing import assert_series_equal
from shapely.geometry import Point, Polygon, box
import geopandas
from geopandas import GeoDataFrame, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.io.file import _detect_driver, _EXTENSION_TO_DRIVER
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas.tests.util import PACKAGE_DIR, validate_boro_df
class FileNumber(object):

    def __init__(self, tmpdir, base, ext):
        self.tmpdir = str(tmpdir)
        self.base = base
        self.ext = ext
        self.fileno = 0

    def __repr__(self):
        filename = '{0}{1:02d}.{2}'.format(self.base, self.fileno, self.ext)
        return os.path.join(self.tmpdir, filename)

    def __next__(self):
        self.fileno += 1
        return repr(self)