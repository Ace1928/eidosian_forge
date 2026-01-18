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
def _decode_metadata(metadata_str):
    """Decode a UTF-8 encoded JSON string to dict

    Parameters
    ----------
    metadata_str : string (UTF-8 encoded)

    Returns
    -------
    dict
    """
    if metadata_str is None:
        return None
    return json.loads(metadata_str.decode('utf-8'))