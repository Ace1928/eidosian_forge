from statistics import mean
import geopandas
from shapely.geometry import LineString
import numpy as np
import pandas as pd
from packaging.version import Version
def _style_highlight(x):
    return {**highlight_kwds}