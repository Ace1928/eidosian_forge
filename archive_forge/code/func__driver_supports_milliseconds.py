import os
from fiona.env import Env
from fiona._env import get_gdal_version_tuple
def _driver_supports_milliseconds(driver):
    """ Returns True if the driver supports milliseconds, False otherwise

        Note: this function is not part of Fiona's public API.
    """
    if _GDAL_VERSION.major < 2:
        return False
    if driver in _drivers_not_supporting_milliseconds:
        if _drivers_not_supporting_milliseconds[driver] is None:
            return False
        elif _drivers_not_supporting_milliseconds[driver] < _GDAL_VERSION:
            return False
    return True