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
def _to_feather(df, path, index=None, compression=None, schema_version=None, **kwargs):
    """
    Write a GeoDataFrame to the Feather format.

    Any geometry columns present are serialized to WKB format in the file.

    Requires 'pyarrow' >= 0.17.

    This is tracking version 1.0.0 of the GeoParquet specification for
    the metadata at: https://github.com/opengeospatial/geoparquet. Writing
    older versions is supported using the `schema_version` keyword.

    .. versionadded:: 0.8

    Parameters
    ----------
    path : str, path object
    index : bool, default None
        If ``True``, always include the dataframe's index(es) as columns
        in the file output.
        If ``False``, the index(es) will not be written to the file.
        If ``None``, the index(ex) will be included as columns in the file
        output except `RangeIndex` which is stored as metadata only.
    compression : {'zstd', 'lz4', 'uncompressed'}, optional
        Name of the compression to use. Use ``"uncompressed"`` for no
        compression. By default uses LZ4 if available, otherwise uncompressed.
    schema_version : {'0.1.0', '0.4.0', '1.0.0', None}
        GeoParquet specification version for the metadata; if not provided
        will default to latest supported version.
    kwargs
        Additional keyword arguments passed to pyarrow.feather.write_feather().
    """
    feather = import_optional_dependency('pyarrow.feather', extra='pyarrow is required for Feather support.')
    import pyarrow
    if Version(pyarrow.__version__) < Version('0.17.0'):
        raise ImportError('pyarrow >= 0.17 required for Feather support')
    if kwargs and 'version' in kwargs and (kwargs['version'] is not None):
        if schema_version is None and kwargs['version'] in SUPPORTED_VERSIONS:
            warnings.warn('the `version` parameter has been replaced with `schema_version`. `version` will instead be passed directly to the underlying feather writer unless `version` is 0.1.0 or 0.4.0.', FutureWarning, stacklevel=2)
            schema_version = kwargs.pop('version')
    path = _expand_user(path)
    table = _geopandas_to_arrow(df, index=index, schema_version=schema_version)
    feather.write_feather(table, path, compression=compression, **kwargs)