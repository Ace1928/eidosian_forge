from packaging.version import Version
import json
import warnings
import numpy as np
from pandas import DataFrame, Series
import geopandas._compat as compat
import shapely
from geopandas._compat import import_optional_dependency
from geopandas.array import from_wkb
from geopandas import GeoDataFrame
import geopandas
from .file import _expand_user
def _validate_dataframe(df):
    """Validate that the GeoDataFrame conforms to requirements for writing
    to Parquet format.

    Raises `ValueError` if the GeoDataFrame is not valid.

    copied from `pandas.io.parquet`

    Parameters
    ----------
    df : GeoDataFrame
    """
    if not isinstance(df, DataFrame):
        raise ValueError('Writing to Parquet/Feather only supports IO with DataFrames')
    if df.columns.inferred_type not in {'string', 'unicode', 'empty'}:
        raise ValueError('Writing to Parquet/Feather requires string column names')
    valid_names = all((isinstance(name, str) for name in df.index.names if name is not None))
    if not valid_names:
        raise ValueError('Index level names must be strings')