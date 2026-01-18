from os.path import join
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image
from .utils import check_integrity, download_and_extract_archive, list_dir, list_files
from .vision import VisionDataset
def _get_target_folder(self) -> str:
    return 'images_background' if self.background else 'images_evaluation'