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
class SceneFlowStereo(StereoMatchingDataset):
    """Dataset interface for `Scene Flow <https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html>`_ datasets.
    This interface provides access to the `FlyingThings3D, `Monkaa` and `Driving` datasets.

    The dataset is expected to have the following structure: ::

        root
            SceneFlow
                Monkaa
                    frames_cleanpass
                        scene1
                            left
                                img1.png
                                img2.png
                            right
                                img1.png
                                img2.png
                        scene2
                            left
                                img1.png
                                img2.png
                            right
                                img1.png
                                img2.png
                    frames_finalpass
                        scene1
                            left
                                img1.png
                                img2.png
                            right
                                img1.png
                                img2.png
                        ...
                        ...
                    disparity
                        scene1
                            left
                                img1.pfm
                                img2.pfm
                            right
                                img1.pfm
                                img2.pfm
                FlyingThings3D
                    ...
                    ...

    Args:
        root (string): Root directory where SceneFlow is located.
        variant (string): Which dataset variant to user, "FlyingThings3D" (default), "Monkaa" or "Driving".
        pass_name (string): Which pass to use, "clean" (default), "final" or "both".
        transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version.

    """

    def __init__(self, root: str, variant: str='FlyingThings3D', pass_name: str='clean', transforms: Optional[Callable]=None) -> None:
        super().__init__(root, transforms)
        root = Path(root) / 'SceneFlow'
        verify_str_arg(variant, 'variant', valid_values=('FlyingThings3D', 'Driving', 'Monkaa'))
        verify_str_arg(pass_name, 'pass_name', valid_values=('clean', 'final', 'both'))
        passes = {'clean': ['frames_cleanpass'], 'final': ['frames_finalpass'], 'both': ['frames_cleanpass', 'frames_finalpass']}[pass_name]
        root = root / variant
        prefix_directories = {'Monkaa': Path('*'), 'FlyingThings3D': Path('*') / '*' / '*', 'Driving': Path('*') / '*' / '*'}
        for p in passes:
            left_image_pattern = str(root / p / prefix_directories[variant] / 'left' / '*.png')
            right_image_pattern = str(root / p / prefix_directories[variant] / 'right' / '*.png')
            self._images += self._scan_pairs(left_image_pattern, right_image_pattern)
            left_disparity_pattern = str(root / 'disparity' / prefix_directories[variant] / 'left' / '*.pfm')
            right_disparity_pattern = str(root / 'disparity' / prefix_directories[variant] / 'right' / '*.pfm')
            self._disparities += self._scan_pairs(left_disparity_pattern, right_disparity_pattern)

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