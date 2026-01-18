import io
from pathlib import Path
import warnings
import numpy as np
from cartopy import config
import cartopy.crs as ccrs
from cartopy.io import Downloader, LocatedImage, RasterSource, fh_getter
class SRTM1Source(_SRTMSource):
    """
    A source of SRTM1 data, which implements Cartopy's :ref:`RasterSource
    interface <raster-source-interface>`.

    """

    def __init__(self, downloader=None, max_nx=3, max_ny=3):
        """
        Parameters
        ----------
        downloader: :class:`cartopy.io.Downloader` instance or None
            The downloader to use for the SRTM1 dataset. If None, the
            downloader will be taken using
            :class:`cartopy.io.Downloader.from_config` with ('SRTM', 'SRTM1')
            as the target.
        max_nx
            The maximum number of x tiles to be combined when
            producing a wider composite for this RasterSource.
        max_ny
            The maximum number of y tiles to be combined when
            producing a taller composite for this RasterSource.

        """
        super().__init__(resolution=1, downloader=downloader, max_nx=max_nx, max_ny=max_ny)