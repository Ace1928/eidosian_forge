import io
import itertools
from pathlib import Path
from urllib.error import HTTPError
import shapefile
import shapely.geometry as sgeom
from cartopy import config
from cartopy.io import Downloader
def acquire_resource(self, target_path, format_dict):
    """
        Download the zip file and extracts the files listed in
        :meth:`zip_file_contents` to the target path.

        Note
        ----
            Because some of the GSHSS data is available with the cartopy
            repository, scales of "l" or "c" will not be downloaded if they
            exist in the ``cartopy.config['repo_data_dir']`` directory.

        """
    repo_fname_pattern = str(Path('shapefiles') / 'gshhs' / '{scale}' / 'GSHHS_{scale}_L?.shp')
    repo_fname_pattern = repo_fname_pattern.format(**format_dict)
    repo_fnames = list(config['repo_data_dir'].glob(repo_fname_pattern))
    if repo_fnames:
        assert len(repo_fnames) == 1, '>1 repo files found for GSHHS'
        return repo_fnames[0]
    self.acquire_all_resources(format_dict)
    if not target_path.exists():
        raise RuntimeError(f'Failed to download and extract GSHHS shapefile to {target_path!r}.')
    return target_path