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
def _ensure_arrow_fs(filesystem):
    """
    Simplified version of pyarrow.fs._ensure_filesystem. This is only needed
    below because `pyarrow.parquet.read_metadata` does not yet accept a
    filesystem keyword (https://issues.apache.org/jira/browse/ARROW-16719)
    """
    from pyarrow import fs
    if isinstance(filesystem, fs.FileSystem):
        return filesystem
    try:
        import fsspec
    except ImportError:
        pass
    else:
        if isinstance(filesystem, fsspec.AbstractFileSystem):
            return fs.PyFileSystem(fs.FSSpecHandler(filesystem))
    return filesystem