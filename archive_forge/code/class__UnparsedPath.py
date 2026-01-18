import pathlib
import re
import sys
from urllib.parse import urlparse
import attr
from rasterio.errors import PathError
@attr.s(slots=True)
class _UnparsedPath(_Path):
    """Encapsulates legacy GDAL filenames

    Attributes
    ----------
    path : str
        The legacy GDAL filename.
    """
    path = attr.ib()

    @property
    def name(self):
        """The unparsed path's original path"""
        return self.path