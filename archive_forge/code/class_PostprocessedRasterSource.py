import collections
from pathlib import Path
import string
from urllib.request import urlopen
import warnings
from cartopy import config
class PostprocessedRasterSource(RasterSourceContainer):
    """
    A :class:`RasterSource` which wraps another, an then applies a
    post-processing step on the raster fetched from the contained source.

    """

    def __init__(self, contained_source, img_post_process):
        """
        Parameters
        ----------
        contained_source: :class:`RasterSource` instance.
            The source of the raster that this container is wrapping.
        img_post_process: callable
            Called after each `fetch_raster` call which yields a non-None
            image result. The callable must accept the :class:`LocatedImage`
            from the contained fetch_raster as its only argument, and must
            return a single LocatedImage.

        """
        super().__init__(contained_source)
        self._post_fetch_fn = img_post_process

    def fetch_raster(self, *args, **kwargs):
        fetch_raster = super().fetch_raster
        located_imgs = fetch_raster(*args, **kwargs)
        if located_imgs:
            located_imgs = [self._post_fetch_fn(img) for img in located_imgs]
        return located_imgs