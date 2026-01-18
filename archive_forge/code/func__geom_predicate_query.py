from typing import Optional
import warnings
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from geopandas import _compat as compat
from geopandas.array import _check_crs, _crs_mismatch_warn
def _geom_predicate_query(left_df, right_df, predicate):
    """Compute geometric comparisons and get matching indices.

    Parameters
    ----------
    left_df : GeoDataFrame
    right_df : GeoDataFrame
    predicate : string
        Binary predicate to query.

    Returns
    -------
    DataFrame
        DataFrame with matching indices in
        columns named `_key_left` and `_key_right`.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Generated spatial index is empty', FutureWarning)
        original_predicate = predicate
        if predicate == 'within':
            predicate = 'contains'
            sindex = left_df.sindex
            input_geoms = right_df.geometry
        else:
            sindex = right_df.sindex
            input_geoms = left_df.geometry
    if sindex:
        l_idx, r_idx = sindex.query(input_geoms, predicate=predicate, sort=False)
        indices = pd.DataFrame({'_key_left': l_idx, '_key_right': r_idx})
    else:
        indices = pd.DataFrame(columns=['_key_left', '_key_right'], dtype=float)
    if original_predicate == 'within':
        indices = indices.rename(columns={'_key_left': '_key_right', '_key_right': '_key_left'})
    return indices