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
class Kitti2012Stereo(StereoMatchingDataset):
    """
    KITTI dataset from the `2012 stereo evaluation benchmark <http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php>`_.
    Uses the RGB images for consistency with KITTI 2015.

    The dataset is expected to have the following structure: ::

        root
            Kitti2012
                testing
                    colored_0
                        1_10.png
                        2_10.png
                        ...
                    colored_1
                        1_10.png
                        2_10.png
                        ...
                training
                    colored_0
                        1_10.png
                        2_10.png
                        ...
                    colored_1
                        1_10.png
                        2_10.png
                        ...
                    disp_noc
                        1.png
                        2.png
                        ...
                    calib

    Args:
        root (string): Root directory where `Kitti2012` is located.
        split (string, optional): The dataset split of scenes, either "train" (default) or "test".
        transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version.
    """
    _has_built_in_disparity_mask = True

    def __init__(self, root: str, split: str='train', transforms: Optional[Callable]=None) -> None:
        super().__init__(root, transforms)
        verify_str_arg(split, 'split', valid_values=('train', 'test'))
        root = Path(root) / 'Kitti2012' / (split + 'ing')
        left_img_pattern = str(root / 'colored_0' / '*_10.png')
        right_img_pattern = str(root / 'colored_1' / '*_10.png')
        self._images = self._scan_pairs(left_img_pattern, right_img_pattern)
        if split == 'train':
            disparity_pattern = str(root / 'disp_noc' / '*.png')
            self._disparities = self._scan_pairs(disparity_pattern, None)
        else:
            self._disparities = list(((None, None) for _ in self._images))

    def _read_disparity(self, file_path: str) -> Tuple[Optional[np.ndarray], None]:
        if file_path is None:
            return (None, None)
        disparity_map = np.asarray(Image.open(file_path)) / 256.0
        disparity_map = disparity_map[None, :, :]
        valid_mask = None
        return (disparity_map, valid_mask)

    def __getitem__(self, index: int) -> T1:
        """Return example at given index.

        Args:
            index(int): The index of the example to retrieve

        Returns:
            tuple: A 4-tuple with ``(img_left, img_right, disparity, valid_mask)``.
            The disparity is a numpy array of shape (1, H, W) and the images are PIL images.
            ``valid_mask`` is implicitly ``None`` if the ``transforms`` parameter does not
            generate a valid mask.
            Both ``disparity`` and ``valid_mask`` are ``None`` if the dataset split is test.
        """
        return cast(T1, super().__getitem__(index))