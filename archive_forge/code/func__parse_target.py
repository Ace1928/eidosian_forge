import csv
import os
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image
from .utils import download_and_extract_archive
from .vision import VisionDataset
def _parse_target(self, index: int) -> List:
    target = []
    with open(self.targets[index]) as inp:
        content = csv.reader(inp, delimiter=' ')
        for line in content:
            target.append({'type': line[0], 'truncated': float(line[1]), 'occluded': int(line[2]), 'alpha': float(line[3]), 'bbox': [float(x) for x in line[4:8]], 'dimensions': [float(x) for x in line[8:11]], 'location': [float(x) for x in line[11:14]], 'rotation_y': float(line[14])})
    return target