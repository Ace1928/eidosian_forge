from click import FileError
class RasterioIOError(OSError):
    """Raised when a dataset cannot be opened using one of the
    registered format drivers."""