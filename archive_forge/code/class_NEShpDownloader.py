import io
import itertools
from pathlib import Path
from urllib.error import HTTPError
import shapefile
import shapely.geometry as sgeom
from cartopy import config
from cartopy.io import Downloader
class NEShpDownloader(Downloader):
    """
    Specialise :class:`cartopy.io.Downloader` to download the zipped
    Natural Earth shapefiles and extract them to the defined location
    (typically user configurable).

    The keys which should be passed through when using the ``format_dict``
    are typically ``category``, ``resolution`` and ``name``.

    """
    FORMAT_KEYS = ('config', 'resolution', 'category', 'name')
    _NE_URL_TEMPLATE = 'https://naturalearth.s3.amazonaws.com/{resolution}_{category}/ne_{resolution}_{name}.zip'

    def __init__(self, url_template=_NE_URL_TEMPLATE, target_path_template=None, pre_downloaded_path_template=''):
        Downloader.__init__(self, url_template, target_path_template, pre_downloaded_path_template)

    def zip_file_contents(self, format_dict):
        """
        Return a generator of the filenames to be found in the downloaded
        natural earth zip file.

        """
        for ext in ['.shp', '.dbf', '.shx', '.prj', '.cpg']:
            yield 'ne_{resolution}_{name}{extension}'.format(extension=ext, **format_dict)

    def acquire_resource(self, target_path, format_dict):
        """
        Download the zip file and extracts the files listed in
        :meth:`zip_file_contents` to the target path.

        """
        from zipfile import ZipFile
        target_dir = Path(target_path).parent
        target_dir.mkdir(parents=True, exist_ok=True)
        url = self.url(format_dict)
        shapefile_online = self._urlopen(url)
        zfh = ZipFile(io.BytesIO(shapefile_online.read()), 'r')
        for member_path in self.zip_file_contents(format_dict):
            member = zfh.getinfo(member_path.replace('\\', '/'))
            with open(target_path.with_suffix(Path(member_path).suffix), 'wb') as fh:
                fh.write(zfh.open(member).read())
        shapefile_online.close()
        zfh.close()
        return target_path

    @staticmethod
    def default_downloader():
        """
        Return a generic, standard, NEShpDownloader instance.

        Typically, a user will not need to call this staticmethod.

        To find the path template of the NEShpDownloader:

            >>> ne_dnldr = NEShpDownloader.default_downloader()
            >>> print(ne_dnldr.target_path_template)
            {config[data_dir]}/shapefiles/natural_earth/{category}/ne_{resolution}_{name}.shp

        """
        default_spec = ('shapefiles', 'natural_earth', '{category}', 'ne_{resolution}_{name}.shp')
        ne_path_template = str(Path('{config[data_dir]}').joinpath(*default_spec))
        pre_path_template = str(Path('{config[pre_existing_data_dir]}').joinpath(*default_spec))
        return NEShpDownloader(target_path_template=ne_path_template, pre_downloaded_path_template=pre_path_template)