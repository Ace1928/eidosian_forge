from statistics import mean
import geopandas
from shapely.geometry import LineString
import numpy as np
import pandas as pd
from packaging.version import Version
def _tooltip_popup(type, fields, gdf, **kwds):
    """get tooltip or popup"""
    import folium
    if fields is False or fields is None or fields == 0:
        return None
    elif fields is True:
        fields = gdf.columns.drop(gdf.geometry.name).to_list()
    elif isinstance(fields, int):
        fields = gdf.columns.drop(gdf.geometry.name).to_list()[:fields]
    elif isinstance(fields, str):
        fields = [fields]
    for field in ['__plottable_column', '__folium_color']:
        if field in fields:
            fields.remove(field)
    fields = list(map(str, fields))
    if type == 'tooltip':
        return folium.GeoJsonTooltip(fields, **kwds)
    elif type == 'popup':
        return folium.GeoJsonPopup(fields, **kwds)