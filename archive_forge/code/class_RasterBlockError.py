from click import FileError
class RasterBlockError(RasterioError):
    """Raised when raster block access fails"""