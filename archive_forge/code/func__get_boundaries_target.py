import os
import shutil
from typing import Any, Callable, Optional, Tuple
import numpy as np
from PIL import Image
from .utils import download_and_extract_archive, download_url, verify_str_arg
from .vision import VisionDataset
def _get_boundaries_target(self, filepath: str) -> np.ndarray:
    mat = self._loadmat(filepath)
    return np.concatenate([np.expand_dims(mat['GTcls'][0]['Boundaries'][0][i][0].toarray(), axis=0) for i in range(self.num_classes)], axis=0)