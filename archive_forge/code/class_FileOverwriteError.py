from click import FileError
class FileOverwriteError(FileError):
    """Raised when Rasterio's CLI refuses to clobber output files."""

    def __init__(self, message):
        """Raise FileOverwriteError with message as hint."""
        super().__init__('', hint=message)