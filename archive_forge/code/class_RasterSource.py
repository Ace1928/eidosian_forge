import collections
from pathlib import Path
import string
from urllib.request import urlopen
import warnings
from cartopy import config
class RasterSource:
    """
    Define the cartopy raster fetching interface.

    A :class:`RasterSource` instance is able to supply images and
    associated extents (as a sequence of :class:`LocatedImage` instances)
    through its :meth:`~RasterSource.fetch_raster` method.

    As a result, further interfacing classes, such as
    :class:`cartopy.mpl.slippy_image_artist.SlippyImageArtist`, can then
    make use of the interface for functionality such as interactive image
    retrieval with pan and zoom functionality.

    .. _raster-source-interface:

    """

    def validate_projection(self, projection):
        """
        Raise an error if this raster source cannot provide images in the
        specified projection.

        Parameters
        ----------
        projection: :class:`cartopy.crs.Projection`
            The desired projection of the image.

        """
        raise NotImplementedError()

    def fetch_raster(self, projection, extent, target_resolution):
        """
        Return a sequence of images with extents given some constraining
        information.

        Parameters
        ----------
        projection: :class:`cartopy.crs.Projection`
            The desired projection of the image.
        extent: iterable of length 4
            The extent of the requested image in projected coordinates.
            The resulting image may not be defined exactly by these extents,
            and so the extent of the resulting image is also returned. The
            extents must be defined in the form
            ``(min_x, max_x, min_y, max_y)``.
        target_resolution: iterable of length 2
            The desired resolution of the image as ``(width, height)`` in
            pixels.

        Returns
        -------
        images
            A sequence of :class:`LocatedImage` instances.

        """
        raise NotImplementedError()