import threading
from shapely.lib import _setup_signal_checks, GEOSException, ShapelyError  # NOQA
class DimensionError(ShapelyError):
    """An error in the number of coordinate dimensions."""