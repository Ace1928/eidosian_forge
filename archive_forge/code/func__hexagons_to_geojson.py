from plotly.express._core import build_dataframe
from plotly.express._doc import make_docstring
from plotly.express._chart_types import choropleth_mapbox, scatter_mapbox
import numpy as np
import pandas as pd
def _hexagons_to_geojson(hexagons_lats, hexagons_lons, ids=None):
    """
    Creates a geojson of hexagonal features based on the outputs of
    _compute_wgs84_hexbin
    """
    features = []
    if ids is None:
        ids = np.arange(len(hexagons_lats))
    for lat, lon, idx in zip(hexagons_lats, hexagons_lons, ids):
        points = np.array([lon, lat]).T.tolist()
        points.append(points[0])
        features.append(dict(type='Feature', id=idx, geometry=dict(type='Polygon', coordinates=[points])))
    return dict(type='FeatureCollection', features=features)