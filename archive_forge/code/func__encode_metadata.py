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
def _encode_metadata(metadata):
    """Encode metadata dict to UTF-8 JSON string

    Parameters
    ----------
    metadata : dict

    Returns
    -------
    UTF-8 encoded JSON string
    """
    return json.dumps(metadata).encode('utf-8')