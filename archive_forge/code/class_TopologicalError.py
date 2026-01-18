import threading
from shapely.lib import _setup_signal_checks, GEOSException, ShapelyError  # NOQA
class TopologicalError(ShapelyError):
    """A geometry is invalid or topologically incorrect."""