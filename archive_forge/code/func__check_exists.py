import csv
import os
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image
from .utils import download_and_extract_archive
from .vision import VisionDataset
def _check_exists(self) -> bool:
    """Check if the data directory exists."""
    folders = [self.image_dir_name]
    if self.train:
        folders.append(self.labels_dir_name)
    return all((os.path.isdir(os.path.join(self._raw_folder, self._location, fname)) for fname in folders))