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
def _create_metadata(df, schema_version=None):
    """Create and encode geo metadata dict.

    Parameters
    ----------
    df : GeoDataFrame
    schema_version : {'0.1.0', '0.4.0', '1.0.0-beta.1', '1.0.0', None}
        GeoParquet specification version; if not provided will default to
        latest supported version.

    Returns
    -------
    dict
    """
    schema_version = schema_version or METADATA_VERSION
    if schema_version not in SUPPORTED_VERSIONS:
        raise ValueError(f'schema_version must be one of: {', '.join(SUPPORTED_VERSIONS)}')
    column_metadata = {}
    for col in df.columns[df.dtypes == 'geometry']:
        series = df[col]
        geometry_types = sorted(Series(series.geom_type.unique()).dropna())
        if schema_version[0] == '0':
            geometry_types_name = 'geometry_type'
            if len(geometry_types) == 1:
                geometry_types = geometry_types[0]
        else:
            geometry_types_name = 'geometry_types'
        crs = None
        if series.crs:
            if schema_version == '0.1.0':
                crs = series.crs.to_wkt()
            else:
                crs = series.crs.to_json_dict()
                _remove_id_from_member_of_ensembles(crs)
        column_metadata[col] = {'encoding': 'WKB', 'crs': crs, geometry_types_name: geometry_types}
        bbox = series.total_bounds.tolist()
        if np.isfinite(bbox).all():
            column_metadata[col]['bbox'] = bbox
    return {'primary_column': df._geometry_column_name, 'columns': column_metadata, 'version': schema_version or METADATA_VERSION, 'creator': {'library': 'geopandas', 'version': geopandas.__version__}}