import warnings
from functools import reduce
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import _check_crs, _crs_mismatch_warn
def _overlay_union(df1, df2):
    """
    Overlay Union operation used in overlay function
    """
    dfinter = _overlay_intersection(df1, df2)
    dfsym = _overlay_symmetric_diff(df1, df2)
    dfunion = pd.concat([dfinter, dfsym], ignore_index=True, sort=False)
    columns = list(dfunion.columns)
    columns.remove('geometry')
    columns.append('geometry')
    return dfunion.reindex(columns=columns)