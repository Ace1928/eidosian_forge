import warnings
from contextlib import contextmanager
import pandas as pd
import shapely
import shapely.wkb
from geopandas import GeoDataFrame
from geopandas import _compat as compat
def _get_srid_from_crs(gdf):
    """
    Get EPSG code from CRS if available. If not, return -1.
    """
    srid = None
    warning_msg = 'Could not parse CRS from the GeoDataFrame. Inserting data without defined CRS.'
    if gdf.crs is not None:
        try:
            for confidence in (100, 70, 25):
                srid = gdf.crs.to_epsg(min_confidence=confidence)
                if srid is not None:
                    break
                auth_srid = gdf.crs.to_authority(auth_name='ESRI', min_confidence=confidence)
                if auth_srid is not None:
                    srid = int(auth_srid[1])
                    break
        except Exception:
            warnings.warn(warning_msg, UserWarning, stacklevel=2)
    if srid is None:
        srid = -1
        warnings.warn(warning_msg, UserWarning, stacklevel=2)
    return srid