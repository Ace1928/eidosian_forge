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
class StereoMatchingDataset(ABC, VisionDataset):
    """Base interface for Stereo matching datasets"""
    _has_built_in_disparity_mask = False

    def __init__(self, root: str, transforms: Optional[Callable]=None) -> None:
        """
        Args:
            root(str): Root directory of the dataset.
            transforms(callable, optional): A function/transform that takes in Tuples of
                (images, disparities, valid_masks) and returns a transformed version of each of them.
                images is a Tuple of (``PIL.Image``, ``PIL.Image``)
                disparities is a Tuple of (``np.ndarray``, ``np.ndarray``) with shape (1, H, W)
                valid_masks is a Tuple of (``np.ndarray``, ``np.ndarray``) with shape (H, W)
                In some cases, when a dataset does not provide disparities, the ``disparities`` and
                ``valid_masks`` can be Tuples containing None values.
                For training splits generally the datasets provide a minimal guarantee of
                images: (``PIL.Image``, ``PIL.Image``)
                disparities: (``np.ndarray``, ``None``) with shape (1, H, W)
                Optionally, based on the dataset, it can return a ``mask`` as well:
                valid_masks: (``np.ndarray | None``, ``None``) with shape (H, W)
                For some test splits, the datasets provides outputs that look like:
                imgaes: (``PIL.Image``, ``PIL.Image``)
                disparities: (``None``, ``None``)
                Optionally, based on the dataset, it can return a ``mask`` as well:
                valid_masks: (``None``, ``None``)
        """
        super().__init__(root=root)
        self.transforms = transforms
        self._images = []
        self._disparities = []

    def _read_img(self, file_path: Union[str, Path]) -> Image.Image:
        img = Image.open(file_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def _scan_pairs(self, paths_left_pattern: str, paths_right_pattern: Optional[str]=None) -> List[Tuple[str, Optional[str]]]:
        left_paths = list(sorted(glob(paths_left_pattern)))
        right_paths: List[Union[None, str]]
        if paths_right_pattern:
            right_paths = list(sorted(glob(paths_right_pattern)))
        else:
            right_paths = list((None for _ in left_paths))
        if not left_paths:
            raise FileNotFoundError(f'Could not find any files matching the patterns: {paths_left_pattern}')
        if not right_paths:
            raise FileNotFoundError(f'Could not find any files matching the patterns: {paths_right_pattern}')
        if len(left_paths) != len(right_paths):
            raise ValueError(f'Found {len(left_paths)} left files but {len(right_paths)} right files using:\n left pattern: {paths_left_pattern}\nright pattern: {paths_right_pattern}\n')
        paths = list(((left, right) for left, right in zip(left_paths, right_paths)))
        return paths

    @abstractmethod
    def _read_disparity(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        pass

    def __getitem__(self, index: int) -> Union[T1, T2]:
        """Return example at given index.

        Args:
            index(int): The index of the example to retrieve

        Returns:
            tuple: A 3 or 4-tuple with ``(img_left, img_right, disparity, Optional[valid_mask])`` where ``valid_mask``
                can be a numpy boolean mask of shape (H, W) if the dataset provides a file
                indicating which disparity pixels are valid. The disparity is a numpy array of
                shape (1, H, W) and the images are PIL images. ``disparity`` is None for
                datasets on which for ``split="test"`` the authors did not provide annotations.
        """
        img_left = self._read_img(self._images[index][0])
        img_right = self._read_img(self._images[index][1])
        dsp_map_left, valid_mask_left = self._read_disparity(self._disparities[index][0])
        dsp_map_right, valid_mask_right = self._read_disparity(self._disparities[index][1])
        imgs = (img_left, img_right)
        dsp_maps = (dsp_map_left, dsp_map_right)
        valid_masks = (valid_mask_left, valid_mask_right)
        if self.transforms is not None:
            imgs, dsp_maps, valid_masks = self.transforms(imgs, dsp_maps, valid_masks)
        if self._has_built_in_disparity_mask or valid_masks[0] is not None:
            return (imgs[0], imgs[1], dsp_maps[0], cast(np.ndarray, valid_masks[0]))
        else:
            return (imgs[0], imgs[1], dsp_maps[0])

    def __len__(self) -> int:
        return len(self._images)