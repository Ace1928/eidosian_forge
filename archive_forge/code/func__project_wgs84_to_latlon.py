from plotly.express._core import build_dataframe
from plotly.express._doc import make_docstring
from plotly.express._chart_types import choropleth_mapbox, scatter_mapbox
import numpy as np
import pandas as pd
def _project_wgs84_to_latlon(x, y):
    """
    Projects WGS84 to lat and lon, used to get regular hexagons on a mapbox map
    """
    lon = x * 180 / np.pi
    lat = (2 * np.arctan(np.exp(y)) - np.pi / 2) * 180 / np.pi
    return (lat, lon)