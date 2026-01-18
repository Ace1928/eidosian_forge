from pyproj.enums import BaseEnum
class Ellipsoidal3DCSAxis(BaseEnum):
    """
    .. versionadded:: 2.5.0

    Ellipsoidal 3D Coordinate System Axis for creating axis with
    with :class:`pyproj.crs.coordinate_system.Ellipsoidal3DCS`

    Attributes
    ----------
    LONGITUDE_LATITUDE_HEIGHT
    LATITUDE_LONGITUDE_HEIGHT
    """
    LONGITUDE_LATITUDE_HEIGHT = 'LONGITUDE_LATITUDE_HEIGHT'
    LATITUDE_LONGITUDE_HEIGHT = 'LATITUDE_LONGITUDE_HEIGHT'