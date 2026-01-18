import io
import numpy as np
import os
import pandas as pd
import warnings
from math import log, floor
from numbers import Number
from plotly import optional_imports
import plotly.colors as clrs
from plotly.figure_factory import utils
from plotly.exceptions import PlotlyError
import plotly.graph_objs as go
def _calculations(df, fips, values, index, f, simplify_county, level, x_centroids, y_centroids, centroid_text, x_traces, y_traces, fips_polygon_map):
    padded_f = str(f).zfill(5)
    if fips_polygon_map[f].type == 'Polygon':
        x = fips_polygon_map[f].simplify(simplify_county).exterior.xy[0].tolist()
        y = fips_polygon_map[f].simplify(simplify_county).exterior.xy[1].tolist()
        x_c, y_c = fips_polygon_map[f].centroid.xy
        county_name_str = str(df[df['FIPS'] == f]['COUNTY_NAME'].iloc[0])
        state_name_str = str(df[df['FIPS'] == f]['STATE_NAME'].iloc[0])
        t_c = 'County: ' + county_name_str + '<br>' + 'State: ' + state_name_str + '<br>' + 'FIPS: ' + padded_f + '<br>Value: ' + str(values[index])
        x_centroids.append(x_c[0])
        y_centroids.append(y_c[0])
        centroid_text.append(t_c)
        x_traces[level] = x_traces[level] + x + [np.nan]
        y_traces[level] = y_traces[level] + y + [np.nan]
    elif fips_polygon_map[f].type == 'MultiPolygon':
        x = [poly.simplify(simplify_county).exterior.xy[0].tolist() for poly in fips_polygon_map[f].geoms]
        y = [poly.simplify(simplify_county).exterior.xy[1].tolist() for poly in fips_polygon_map[f].geoms]
        x_c = [poly.centroid.xy[0].tolist() for poly in fips_polygon_map[f].geoms]
        y_c = [poly.centroid.xy[1].tolist() for poly in fips_polygon_map[f].geoms]
        county_name_str = str(df[df['FIPS'] == f]['COUNTY_NAME'].iloc[0])
        state_name_str = str(df[df['FIPS'] == f]['STATE_NAME'].iloc[0])
        text = 'County: ' + county_name_str + '<br>' + 'State: ' + state_name_str + '<br>' + 'FIPS: ' + padded_f + '<br>Value: ' + str(values[index])
        t_c = [text for poly in fips_polygon_map[f].geoms]
        x_centroids = x_c + x_centroids
        y_centroids = y_c + y_centroids
        centroid_text = t_c + centroid_text
        for x_y_idx in range(len(x)):
            x_traces[level] = x_traces[level] + x[x_y_idx] + [np.nan]
            y_traces[level] = y_traces[level] + y[x_y_idx] + [np.nan]
    return (x_traces, y_traces, x_centroids, y_centroids, centroid_text)