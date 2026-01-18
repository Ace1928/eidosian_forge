import threading
from shapely.lib import _setup_signal_checks, GEOSException, ShapelyError  # NOQA
class EmptyPartError(ShapelyError):
    """An error signifying an empty part was encountered when creating a multi-part."""