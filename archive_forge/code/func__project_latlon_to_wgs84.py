from plotly.express._core import build_dataframe
from plotly.express._doc import make_docstring
from plotly.express._chart_types import choropleth_mapbox, scatter_mapbox
import numpy as np
import pandas as pd
def _project_latlon_to_wgs84(lat, lon):
    """
    Projects lat and lon to WGS84, used to get regular hexagons on a mapbox map
    """
    x = lon * np.pi / 180
    y = np.arctanh(np.sin(lat * np.pi / 180))
    return (x, y)