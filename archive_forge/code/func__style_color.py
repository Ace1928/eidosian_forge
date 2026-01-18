from statistics import mean
import geopandas
from shapely.geometry import LineString
import numpy as np
import pandas as pd
from packaging.version import Version
def _style_color(x):
    base_style = {'fillColor': x['properties'][color], **style_kwds}
    return {**base_style, **style_kwds_function(x)}