import os
from fiona.env import Env
from fiona._env import get_gdal_version_tuple
def _driver_supports_mode(driver, mode):
    """ Returns True if driver supports mode, False otherwise

        Note: this function is not part of Fiona's public API.
    """
    if driver not in supported_drivers:
        return False
    if mode not in supported_drivers[driver]:
        return False
    if driver in driver_mode_mingdal[mode]:
        if _GDAL_VERSION < driver_mode_mingdal[mode][driver]:
            return False
    return True