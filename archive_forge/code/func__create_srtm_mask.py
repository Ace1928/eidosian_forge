import io
from pathlib import Path
import warnings
import numpy as np
from cartopy import config
import cartopy.crs as ccrs
from cartopy.io import Downloader, LocatedImage, RasterSource, fh_getter
@staticmethod
def _create_srtm_mask(resolution, filename=None):
    """
        Return a NumPy mask of available lat/lon.

        This is slow as it must query the SRTM server to identify the
        continent from which the tile comes. Hence a NumPy file with this
        content exists in ``SRTMDownloader._SRTM_LOOKUP_CACHE``.

        The NumPy file was created with::

            import cartopy.io.srtm as srtm
            import numpy as np
            np.savez_compressed(srtm.SRTMDownloader._SRTM_LOOKUP_CACHE,
                                mask=srtm.SRTMDownloader._create_srtm_mask(3))

        """
    from bs4 import BeautifulSoup
    if filename is None:
        from urllib.request import urlopen
        url = SRTMDownloader._SRTM_BASE_URL.format(resolution=resolution)
        with urlopen(url) as f:
            html = f.read()
    else:
        with open(filename) as f:
            html = f.read()
    mask = np.zeros((360, 181), dtype=bool)
    soup = BeautifulSoup(html)
    for link in soup('a'):
        name = str(link.text).strip()
        if name[0] in 'NS' and name.endswith('.hgt.zip'):
            lat = int(name[1:3])
            if name[0] == 'N':
                colat = 90 - lat
            else:
                colat = 90 + lat
            lon = int(name[4:7])
            if name[3] == 'W':
                lon = 360 - lon
            mask[lon, colat] = True
    return mask