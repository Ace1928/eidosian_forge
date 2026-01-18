import threading
from shapely.lib import _setup_signal_checks, GEOSException, ShapelyError  # NOQA
class GeometryTypeError(ShapelyError):
    """
    An error raised when the type of the geometry in question is
    unrecognized or inappropriate.
    """