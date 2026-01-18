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
class ETH3DStereo(StereoMatchingDataset):
    """ETH3D `Low-Res Two-View <https://www.eth3d.net/datasets>`_ dataset.

    The dataset is expected to have the following structure: ::

        root
            ETH3D
                two_view_training
                    scene1
                        im1.png
                        im0.png
                        images.txt
                        cameras.txt
                        calib.txt
                    scene2
                        im1.png
                        im0.png
                        images.txt
                        cameras.txt
                        calib.txt
                    ...
                two_view_training_gt
                    scene1
                        disp0GT.pfm
                        mask0nocc.png
                    scene2
                        disp0GT.pfm
                        mask0nocc.png
                    ...
                two_view_testing
                    scene1
                        im1.png
                        im0.png
                        images.txt
                        cameras.txt
                        calib.txt
                    scene2
                        im1.png
                        im0.png
                        images.txt
                        cameras.txt
                        calib.txt
                    ...

    Args:
        root (string): Root directory of the ETH3D Dataset.
        split (string, optional): The dataset split of scenes, either "train" (default) or "test".
        transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version.
    """
    _has_built_in_disparity_mask = True

    def __init__(self, root: str, split: str='train', transforms: Optional[Callable]=None) -> None:
        super().__init__(root, transforms)
        verify_str_arg(split, 'split', valid_values=('train', 'test'))
        root = Path(root) / 'ETH3D'
        img_dir = 'two_view_training' if split == 'train' else 'two_view_test'
        anot_dir = 'two_view_training_gt'
        left_img_pattern = str(root / img_dir / '*' / 'im0.png')
        right_img_pattern = str(root / img_dir / '*' / 'im1.png')
        self._images = self._scan_pairs(left_img_pattern, right_img_pattern)
        if split == 'test':
            self._disparities = list(((None, None) for _ in self._images))
        else:
            disparity_pattern = str(root / anot_dir / '*' / 'disp0GT.pfm')
            self._disparities = self._scan_pairs(disparity_pattern, None)

    def _read_disparity(self, file_path: str) -> Union[Tuple[None, None], Tuple[np.ndarray, np.ndarray]]:
        if file_path is None:
            return (None, None)
        disparity_map = _read_pfm_file(file_path)
        disparity_map = np.abs(disparity_map)
        mask_path = Path(file_path).parent / 'mask0nocc.png'
        valid_mask = Image.open(mask_path)
        valid_mask = np.asarray(valid_mask).astype(bool)
        return (disparity_map, valid_mask)

    def __getitem__(self, index: int) -> T2:
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
        return cast(T2, super().__getitem__(index))