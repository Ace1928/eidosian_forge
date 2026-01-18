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
def _sha256sum(input_file):
    """
    Return sha256 checksum of file given by path.
    """
    hasher = hashlib.sha256()
    with open(input_file, 'rb') as file:
        for chunk in iter(lambda: file.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()