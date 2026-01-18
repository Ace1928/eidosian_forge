import collections
from pathlib import Path
import string
from urllib.request import urlopen
import warnings
from cartopy import config
class RasterSourceContainer(RasterSource):
    """
    A container which simply calls the appropriate methods on the
    contained :class:`RasterSource`.

    """

    def __init__(self, contained_source):
        """
        Parameters
        ----------
        contained_source: :class:`RasterSource` instance.
            The source of the raster that this container is wrapping.

        """
        self._source = contained_source

    def fetch_raster(self, projection, extent, target_resolution):
        return self._source.fetch_raster(projection, extent, target_resolution)

    def validate_projection(self, projection):
        return self._source.validate_projection(projection)