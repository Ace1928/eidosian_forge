import itertools
import os
from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image
from ..io.image import _read_png_16
from .utils import _read_pfm, verify_str_arg
from .vision import VisionDataset
class KittiFlow(FlowDataset):
    """`KITTI <http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow>`__ dataset for optical flow (2015).

    The dataset is expected to have the following structure: ::

        root
            KittiFlow
                testing
                    image_2
                training
                    image_2
                    flow_occ

    Args:
        root (string): Root directory of the KittiFlow Dataset.
        split (string, optional): The dataset split, either "train" (default) or "test"
        transforms (callable, optional): A function/transform that takes in
            ``img1, img2, flow, valid_flow_mask`` and returns a transformed version.
    """
    _has_builtin_flow_mask = True

    def __init__(self, root: str, split: str='train', transforms: Optional[Callable]=None) -> None:
        super().__init__(root=root, transforms=transforms)
        verify_str_arg(split, 'split', valid_values=('train', 'test'))
        root = Path(root) / 'KittiFlow' / (split + 'ing')
        images1 = sorted(glob(str(root / 'image_2' / '*_10.png')))
        images2 = sorted(glob(str(root / 'image_2' / '*_11.png')))
        if not images1 or not images2:
            raise FileNotFoundError('Could not find the Kitti flow images. Please make sure the directory structure is correct.')
        for img1, img2 in zip(images1, images2):
            self._image_list += [[img1, img2]]
        if split == 'train':
            self._flow_list = sorted(glob(str(root / 'flow_occ' / '*_10.png')))

    def __getitem__(self, index: int) -> Union[T1, T2]:
        """Return example at given index.

        Args:
            index(int): The index of the example to retrieve

        Returns:
            tuple: A 4-tuple with ``(img1, img2, flow, valid_flow_mask)``
            where ``valid_flow_mask`` is a numpy boolean mask of shape (H, W)
            indicating which flow values are valid. The flow is a numpy array of
            shape (2, H, W) and the images are PIL images. ``flow`` and ``valid_flow_mask`` are None if
            ``split="test"``.
        """
        return super().__getitem__(index)

    def _read_flow(self, file_name: str) -> Tuple[np.ndarray, np.ndarray]:
        return _read_16bits_png_with_flow_and_valid_mask(file_name)