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
def _bbox_from_geom(geom: dict[str, Any]) -> Optional[BBox]:
    """
    Get the bounding box from geojson geometry
    """
    if 'coordinates' not in geom or 'type' not in geom:
        return None
    coordinates = geom['coordinates']
    if geom['type'] != 'MultiPolygon':
        return _bbox_from_coords(coordinates)
    found_minus_180 = False
    found_plus_180 = False
    bboxes = []
    for coordinate_set in coordinates:
        bbox = _bbox_from_coords(coordinate_set)
        if bbox is None:
            continue
        if bbox.west == -180:
            found_minus_180 = True
        elif bbox.east == 180:
            found_plus_180 = True
        bboxes.append(bbox)
    grid_bbox = None
    for bbox in bboxes:
        if found_minus_180 and found_plus_180 and (bbox.west == -180):
            bbox.west = 180
            bbox.east += 360
        if grid_bbox is None:
            grid_bbox = bbox
        else:
            grid_bbox.west = min(grid_bbox.west, bbox.west)
            grid_bbox.south = min(grid_bbox.south, bbox.south)
            grid_bbox.north = max(grid_bbox.north, bbox.north)
            grid_bbox.east = max(grid_bbox.east, bbox.east)
    return grid_bbox