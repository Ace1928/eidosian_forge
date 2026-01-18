import os.path
from typing import Any, Callable, cast, Optional, Tuple
import numpy as np
from PIL import Image
from .utils import check_integrity, download_and_extract_archive, verify_str_arg
from .vision import VisionDataset
def __loadfile(self, data_file: str, labels_file: Optional[str]=None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    labels = None
    if labels_file:
        path_to_labels = os.path.join(self.root, self.base_folder, labels_file)
        with open(path_to_labels, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8) - 1
    path_to_data = os.path.join(self.root, self.base_folder, data_file)
    with open(path_to_data, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 1, 3, 2))
    return (images, labels)