import os.path
from typing import Any, Callable, cast, Optional, Tuple
import numpy as np
from PIL import Image
from .utils import check_integrity, download_and_extract_archive, verify_str_arg
from .vision import VisionDataset
def __load_folds(self, folds: Optional[int]) -> None:
    if folds is None:
        return
    path_to_folds = os.path.join(self.root, self.base_folder, self.folds_list_file)
    with open(path_to_folds) as f:
        str_idx = f.read().splitlines()[folds]
        list_idx = np.fromstring(str_idx, dtype=np.int64, sep=' ')
        self.data = self.data[list_idx, :, :, :]
        if self.labels is not None:
            self.labels = self.labels[list_idx]