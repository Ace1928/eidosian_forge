import csv
import os
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image
from .utils import download_and_extract_archive
from .vision import VisionDataset
@property
def _raw_folder(self) -> str:
    return os.path.join(self.root, self.__class__.__name__, 'raw')