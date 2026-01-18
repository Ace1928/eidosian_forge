import functools
import json
import os
import random
import shutil
from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path
from typing import Callable, cast, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
from .utils import _read_pfm, download_and_extract_archive, verify_str_arg
from .vision import VisionDataset
class InStereo2k(StereoMatchingDataset):
    """`InStereo2k <https://github.com/YuhuaXu/StereoDataset>`_ dataset.

    The dataset is expected to have the following structure: ::

        root
            InStereo2k
                train
                    scene1
                        left.png
                        right.png
                        left_disp.png
                        right_disp.png
                        ...
                    scene2
                    ...
                test
                    scene1
                        left.png
                        right.png
                        left_disp.png
                        right_disp.png
                        ...
                    scene2
                    ...

    Args:
        root (string): Root directory where InStereo2k is located.
        split (string): Either "train" or "test".
        transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version.
    """

    def __init__(self, root: str, split: str='train', transforms: Optional[Callable]=None) -> None:
        super().__init__(root, transforms)
        root = Path(root) / 'InStereo2k' / split
        verify_str_arg(split, 'split', valid_values=('train', 'test'))
        left_img_pattern = str(root / '*' / 'left.png')
        right_img_pattern = str(root / '*' / 'right.png')
        self._images = self._scan_pairs(left_img_pattern, right_img_pattern)
        left_disparity_pattern = str(root / '*' / 'left_disp.png')
        right_disparity_pattern = str(root / '*' / 'right_disp.png')
        self._disparities = self._scan_pairs(left_disparity_pattern, right_disparity_pattern)

    def _read_disparity(self, file_path: str) -> Tuple[np.ndarray, None]:
        disparity_map = np.asarray(Image.open(file_path), dtype=np.float32)
        disparity_map = disparity_map[None, :, :] / 1024.0
        valid_mask = None
        return (disparity_map, valid_mask)

    def __getitem__(self, index: int) -> T1:
        """Return example at given index.

        Args:
            index(int): The index of the example to retrieve

        Returns:
            tuple: A 3-tuple with ``(img_left, img_right, disparity)``.
            The disparity is a numpy array of shape (1, H, W) and the images are PIL images.
            If a ``valid_mask`` is generated within the ``transforms`` parameter,
            a 4-tuple with ``(img_left, img_right, disparity, valid_mask)`` is returned.
        """
        return cast(T1, super().__getitem__(index))