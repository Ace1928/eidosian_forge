import importlib
import platform
import sys
def _get_C_info():
    """Information on system PROJ, GDAL, GEOS
    Returns
    -------
    c_info: dict
        system PROJ information
    """
    try:
        import pyproj
        proj_version = pyproj.proj_version_str
    except Exception:
        proj_version = None
    try:
        import pyproj
        proj_dir = pyproj.datadir.get_data_dir()
    except Exception:
        proj_dir = None
    try:
        import shapely._buildcfg
        geos_version = '{}.{}.{}'.format(*shapely._buildcfg.geos_version)
        geos_dir = shapely._buildcfg.geos_library_path
    except Exception:
        try:
            from shapely import geos_version_string
            geos_version = geos_version_string
            geos_dir = None
        except Exception:
            geos_version = None
            geos_dir = None
    try:
        import fiona
        gdal_version = fiona.env.get_gdal_release_name()
    except Exception:
        gdal_version = None
    try:
        import fiona
        gdal_dir = fiona.env.GDALDataFinder().search()
    except Exception:
        gdal_dir = None
    if gdal_version is None:
        try:
            import pyogrio
            gdal_version = pyogrio.__gdal_version_string__
            gdal_dir = None
        except Exception:
            pass
        try:
            from pyogrio import get_gdal_data_path
            gdal_dir = get_gdal_data_path()
        except Exception:
            pass
    blob = [('GEOS', geos_version), ('GEOS lib', geos_dir), ('GDAL', gdal_version), ('GDAL data dir', gdal_dir), ('PROJ', proj_version), ('PROJ data dir', proj_dir)]
    return dict(blob)