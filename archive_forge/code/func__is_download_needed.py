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
def _is_download_needed(grid_name: str) -> bool:
    """
    Run through all of the PROJ directories to see if the
    file already exists.
    """
    if Path(get_user_data_dir(), grid_name).exists():
        return False
    for data_dir in get_data_dir().split(os.pathsep):
        if Path(data_dir, grid_name).exists():
            return False
    return True