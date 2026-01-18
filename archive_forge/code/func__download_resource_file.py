import hashlib
import json
import os
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlretrieve
from pyproj._sync import get_proj_endpoint
from pyproj.aoi import BBox
from pyproj.datadir import get_data_dir, get_user_data_dir
def _download_resource_file(file_url, short_name, directory, verbose=False, sha256=None):
    """
    Download resource file from PROJ url
    """
    if verbose:
        print(f'Downloading: {file_url}')
    tmp_path = Path(directory, f'{short_name}.part')
    try:
        urlretrieve(file_url, tmp_path)
        if sha256 is not None and sha256 != _sha256sum(tmp_path):
            raise RuntimeError(f'SHA256 mismatch: {short_name}')
        tmp_path.replace(Path(directory, short_name))
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass