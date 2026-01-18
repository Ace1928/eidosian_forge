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
def _bbox_from_coords(coords: list) -> Optional[BBox]:
    """
    Get the bounding box from coordinates
    """
    try:
        xxx, yyy = zip(*coords)
        return BBox(west=min(xxx), south=min(yyy), east=max(xxx), north=max(yyy))
    except ValueError:
        pass
    coord_bbox = None
    for coord_set in coords:
        bbox = _bbox_from_coords(coord_set)
        if coord_bbox is None:
            coord_bbox = bbox
        else:
            coord_bbox.west = min(coord_bbox.west, bbox.west)
            coord_bbox.south = min(coord_bbox.south, bbox.south)
            coord_bbox.north = max(coord_bbox.north, bbox.north)
            coord_bbox.east = max(coord_bbox.east, bbox.east)
    return coord_bbox