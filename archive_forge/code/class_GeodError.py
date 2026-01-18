from pyproj._datadir import _clear_proj_error, _get_proj_error
class GeodError(RuntimeError):
    """Raised when a Geod error occurs."""