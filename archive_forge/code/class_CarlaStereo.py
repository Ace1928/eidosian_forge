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
class CarlaStereo(StereoMatchingDataset):
    """
    Carla simulator data linked in the `CREStereo github repo <https://github.com/megvii-research/CREStereo>`_.

    The dataset is expected to have the following structure: ::

        root
            carla-highres
                trainingF
                    scene1
                        img0.png
                        img1.png
                        disp0GT.pfm
                        disp1GT.pfm
                        calib.txt
                    scene2
                        img0.png
                        img1.png
                        disp0GT.pfm
                        disp1GT.pfm
                        calib.txt
                    ...

    Args:
        root (string): Root directory where `carla-highres` is located.
        transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version.
    """

    def __init__(self, root: str, transforms: Optional[Callable]=None) -> None:
        super().__init__(root, transforms)
        root = Path(root) / 'carla-highres'
        left_image_pattern = str(root / 'trainingF' / '*' / 'im0.png')
        right_image_pattern = str(root / 'trainingF' / '*' / 'im1.png')
        imgs = self._scan_pairs(left_image_pattern, right_image_pattern)
        self._images = imgs
        left_disparity_pattern = str(root / 'trainingF' / '*' / 'disp0GT.pfm')
        right_disparity_pattern = str(root / 'trainingF' / '*' / 'disp1GT.pfm')
        disparities = self._scan_pairs(left_disparity_pattern, right_disparity_pattern)
        self._disparities = disparities

    def _read_disparity(self, file_path: str) -> Tuple[np.ndarray, None]:
        disparity_map = _read_pfm_file(file_path)
        disparity_map = np.abs(disparity_map)
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