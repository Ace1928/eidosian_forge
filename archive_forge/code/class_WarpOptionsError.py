from click import FileError
class WarpOptionsError(RasterioError):
    """Raised when options for a warp operation are invalid"""