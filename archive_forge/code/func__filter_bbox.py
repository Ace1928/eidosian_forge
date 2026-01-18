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
def _filter_bbox(feature: dict[str, Any], bbox: BBox, spatial_test: str, include_world_coverage: bool) -> bool:
    """
    Filter by the bounding box. Designed to use with 'filter'
    """
    geom = feature.get('geometry')
    if geom is not None:
        geom_bbox = _bbox_from_geom(geom)
        if geom_bbox is None:
            return False
        if geom_bbox.east - geom_bbox.west > 359 and geom_bbox.north - geom_bbox.south > 179:
            if not include_world_coverage:
                return False
            geom_bbox.west = -float('inf')
            geom_bbox.east = float('inf')
        elif geom_bbox.east > 180 and bbox.west < -180:
            geom_bbox.west -= 360
            geom_bbox.east -= 360
        return getattr(bbox, spatial_test)(geom_bbox)
    return False