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
class CREStereo(StereoMatchingDataset):
    """Synthetic dataset used in training the `CREStereo <https://arxiv.org/pdf/2203.11483.pdf>`_ architecture.
    Dataset details on the official paper `repo <https://github.com/megvii-research/CREStereo>`_.

    The dataset is expected to have the following structure: ::

        root
            CREStereo
                tree
                    img1_left.jpg
                    img1_right.jpg
                    img1_left.disp.jpg
                    img1_right.disp.jpg
                    img2_left.jpg
                    img2_right.jpg
                    img2_left.disp.jpg
                    img2_right.disp.jpg
                    ...
                shapenet
                    img1_left.jpg
                    img1_right.jpg
                    img1_left.disp.jpg
                    img1_right.disp.jpg
                    ...
                reflective
                    img1_left.jpg
                    img1_right.jpg
                    img1_left.disp.jpg
                    img1_right.disp.jpg
                    ...
                hole
                    img1_left.jpg
                    img1_right.jpg
                    img1_left.disp.jpg
                    img1_right.disp.jpg
                    ...

    Args:
        root (str): Root directory of the dataset.
        transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version.
    """
    _has_built_in_disparity_mask = True

    def __init__(self, root: str, transforms: Optional[Callable]=None) -> None:
        super().__init__(root, transforms)
        root = Path(root) / 'CREStereo'
        dirs = ['shapenet', 'reflective', 'tree', 'hole']
        for s in dirs:
            left_image_pattern = str(root / s / '*_left.jpg')
            right_image_pattern = str(root / s / '*_right.jpg')
            imgs = self._scan_pairs(left_image_pattern, right_image_pattern)
            self._images += imgs
            left_disparity_pattern = str(root / s / '*_left.disp.png')
            right_disparity_pattern = str(root / s / '*_right.disp.png')
            disparities = self._scan_pairs(left_disparity_pattern, right_disparity_pattern)
            self._disparities += disparities

    def _read_disparity(self, file_path: str) -> Tuple[np.ndarray, None]:
        disparity_map = np.asarray(Image.open(file_path), dtype=np.float32)
        disparity_map = disparity_map[None, :, :] / 32.0
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
        """
        return cast(T1, super().__getitem__(index))