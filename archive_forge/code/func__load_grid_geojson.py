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
def _load_grid_geojson(target_directory=None) -> dict[str, Any]:
    """
    Returns
    -------
    dict[str, Any]:
        The PROJ grid data list.
    """
    if target_directory is None:
        target_directory = get_user_data_dir(True)
    local_path = Path(target_directory, 'files.geojson')
    if not local_path.exists() or (datetime.utcnow() - datetime.fromtimestamp(local_path.stat().st_mtime)).days > 0:
        _download_resource_file(file_url=f'{get_proj_endpoint()}/files.geojson', short_name='files.geojson', directory=target_directory)
    return json.loads(local_path.read_text(encoding='utf-8'))