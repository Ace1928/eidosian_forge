import threading
from shapely.lib import _setup_signal_checks, GEOSException, ShapelyError  # NOQA
class UnsupportedGEOSVersionError(ShapelyError):
    """Raised when the GEOS library version does not support a certain operation."""