import os
from fiona.env import Env
from fiona._env import get_gdal_version_tuple
def _driver_converts_field_type_silently_to_str(driver, field_type):
    """ Returns True if the driver converts the field_type silently to str, False otherwise

        Note: this function is not part of Fiona's public API.
    """
    if field_type in _driver_converts_to_str and driver in _driver_converts_to_str[field_type]:
        if _driver_converts_to_str[field_type][driver] is None:
            return True
        elif _GDAL_VERSION < _driver_converts_to_str[field_type][driver]:
            return True
    return False