from plotly.express._core import build_dataframe
from plotly.express._doc import make_docstring
from plotly.express._chart_types import choropleth_mapbox, scatter_mapbox
import numpy as np
import pandas as pd
def _compute_wgs84_hexbin(lat=None, lon=None, lat_range=None, lon_range=None, color=None, nx=None, agg_func=None, min_count=None):
    """
    Computes the lat-lon aggregation at hexagonal bin level.
    Latitude and longitude need to be projected to WGS84 before aggregating
    in order to display regular hexagons on the map.

    Parameters
    ----------
    lat : np.ndarray
        Array of latitudes (shape N)
    lon : np.ndarray
        Array of longitudes (shape N)
    lat_range : np.ndarray
        Min and max latitudes (shape 2)
    lon_range : np.ndarray
        Min and max longitudes (shape 2)
    color : np.ndarray
        Metric to aggregate at hexagon level (shape N)
    nx : int
        Number of hexagons horizontally
    agg_func : function
        Numpy compatible aggregator, this function must take a one-dimensional
        np.ndarray as input and output a scalar
    min_count : int
        Minimum number of points in the hexagon for the hexagon to be displayed

    Returns
    -------
    np.ndarray
        Lat coordinates of each hexagon (shape M x 6)
    np.ndarray
        Lon coordinates of each hexagon (shape M x 6)
    pd.Series
        Unique id for each hexagon, to be used in the geojson data (shape M)
    np.ndarray
        Aggregated value in each hexagon (shape M)

    """
    x, y = _project_latlon_to_wgs84(lat, lon)
    if lat_range is None:
        lat_range = np.array([lat.min(), lat.max()])
    if lon_range is None:
        lon_range = np.array([lon.min(), lon.max()])
    x_range, y_range = _project_latlon_to_wgs84(lat_range, lon_range)
    hxs, hys, centers, agreggated_value = _compute_hexbin(x, y, x_range, y_range, color, nx, agg_func, min_count)
    hexagons_lats, hexagons_lons = _project_wgs84_to_latlon(hxs, hys)
    centers = centers.astype(str)
    hexagons_ids = pd.Series(centers[:, 0]) + ',' + pd.Series(centers[:, 1])
    return (hexagons_lats, hexagons_lons, hexagons_ids, agreggated_value)