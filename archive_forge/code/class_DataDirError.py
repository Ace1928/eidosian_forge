from pyproj._datadir import _clear_proj_error, _get_proj_error
class DataDirError(RuntimeError):
    """Raised when a the data directory was not found."""