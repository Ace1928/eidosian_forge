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
class FallingThingsStereo(StereoMatchingDataset):
    """`FallingThings <https://research.nvidia.com/publication/2018-06_falling-things-synthetic-dataset-3d-object-detection-and-pose-estimation>`_ dataset.

    The dataset is expected to have the following structure: ::

        root
            FallingThings
                single
                    dir1
                        scene1
                            _object_settings.json
                            _camera_settings.json
                            image1.left.depth.png
                            image1.right.depth.png
                            image1.left.jpg
                            image1.right.jpg
                            image2.left.depth.png
                            image2.right.depth.png
                            image2.left.jpg
                            image2.right
                            ...
                        scene2
                    ...
                mixed
                    scene1
                        _object_settings.json
                        _camera_settings.json
                        image1.left.depth.png
                        image1.right.depth.png
                        image1.left.jpg
                        image1.right.jpg
                        image2.left.depth.png
                        image2.right.depth.png
                        image2.left.jpg
                        image2.right
                        ...
                    scene2
                    ...

    Args:
        root (string): Root directory where FallingThings is located.
        variant (string): Which variant to use. Either "single", "mixed", or "both".
        transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version.
    """

    def __init__(self, root: str, variant: str='single', transforms: Optional[Callable]=None) -> None:
        super().__init__(root, transforms)
        root = Path(root) / 'FallingThings'
        verify_str_arg(variant, 'variant', valid_values=('single', 'mixed', 'both'))
        variants = {'single': ['single'], 'mixed': ['mixed'], 'both': ['single', 'mixed']}[variant]
        split_prefix = {'single': Path('*') / '*', 'mixed': Path('*')}
        for s in variants:
            left_img_pattern = str(root / s / split_prefix[s] / '*.left.jpg')
            right_img_pattern = str(root / s / split_prefix[s] / '*.right.jpg')
            self._images += self._scan_pairs(left_img_pattern, right_img_pattern)
            left_disparity_pattern = str(root / s / split_prefix[s] / '*.left.depth.png')
            right_disparity_pattern = str(root / s / split_prefix[s] / '*.right.depth.png')
            self._disparities += self._scan_pairs(left_disparity_pattern, right_disparity_pattern)

    def _read_disparity(self, file_path: str) -> Tuple[np.ndarray, None]:
        depth = np.asarray(Image.open(file_path))
        camera_settings_path = Path(file_path).parent / '_camera_settings.json'
        with open(camera_settings_path, 'r') as f:
            intrinsics = json.load(f)
            focal = intrinsics['camera_settings'][0]['intrinsic_settings']['fx']
            baseline, pixel_constant = (6, 100)
            disparity_map = baseline * focal * pixel_constant / depth.astype(np.float32)
            disparity_map = disparity_map[None, :, :]
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