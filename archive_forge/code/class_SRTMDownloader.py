import io
from pathlib import Path
import warnings
import numpy as np
from cartopy import config
import cartopy.crs as ccrs
from cartopy.io import Downloader, LocatedImage, RasterSource, fh_getter
class SRTMDownloader(Downloader):
    """
    Provide a SRTM download mechanism.

    """
    FORMAT_KEYS = ('config', 'resolution', 'x', 'y')
    _SRTM_BASE_URL = 'https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL{resolution}.003/2000.02.11/'
    _SRTM_LOOKUP_CACHE = Path(__file__).parent / 'srtm.npz'
    _SRTM_LOOKUP_MASK = np.load(_SRTM_LOOKUP_CACHE)['mask']
    "\n    The SRTM lookup mask determines whether keys such as 'N43E043' are\n    available to download.\n\n    "

    def __init__(self, target_path_template, pre_downloaded_path_template=''):
        Downloader.__init__(self, None, target_path_template, pre_downloaded_path_template)

    def url(self, format_dict):
        warnings.warn('SRTM requires an account set up and log in to access. Use of this Downloader is likely to fail with HTTP 401 errors.')
        lat = int(format_dict['y'][1:])
        if format_dict['y'][0] == 'N':
            colat = 90 - lat
        else:
            colat = 90 + lat
        lon = int(format_dict['x'][1:4])
        if format_dict['x'][0] == 'W':
            lon = 360 - lon
        if SRTMDownloader._SRTM_LOOKUP_MASK[lon, colat]:
            return (SRTMDownloader._SRTM_BASE_URL + '{y}{x}.SRTMGL{resolution}.hgt.zip').format(**format_dict)
        else:
            return None

    def acquire_resource(self, target_path, format_dict):
        from zipfile import ZipFile
        target_dir = Path(target_path).parent
        target_dir.mkdir(parents=True, exist_ok=True)
        url = self.url(format_dict)
        srtm_online = self._urlopen(url)
        zfh = ZipFile(io.BytesIO(srtm_online.read()), 'r')
        zip_member_path = '{y}{x}.hgt'.format(**format_dict)
        member = zfh.getinfo(zip_member_path)
        with open(target_path, 'wb') as fh:
            fh.write(zfh.open(member).read())
        srtm_online.close()
        zfh.close()
        return target_path

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

    @classmethod
    def default_downloader(cls):
        """
        Return a typical downloader for this class. In general, this static
        method is used to create the default configuration in cartopy.config

        """
        default_spec = ('SRTM', 'SRTMGL{resolution}', '{y}{x}.hgt')
        target_path_template = str(Path('{config[data_dir]}').joinpath(*default_spec))
        pre_path_template = str(Path('{config[pre_existing_data_dir]}').joinpath(*default_spec))
        return cls(target_path_template=target_path_template, pre_downloaded_path_template=pre_path_template)