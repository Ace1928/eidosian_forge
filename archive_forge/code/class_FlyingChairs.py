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
class FlyingChairs(FlowDataset):
    """`FlyingChairs <https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs>`_ Dataset for optical flow.

    You will also need to download the FlyingChairs_train_val.txt file from the dataset page.

    The dataset is expected to have the following structure: ::

        root
            FlyingChairs
                data
                    00001_flow.flo
                    00001_img1.ppm
                    00001_img2.ppm
                    ...
                FlyingChairs_train_val.txt


    Args:
        root (string): Root directory of the FlyingChairs Dataset.
        split (string, optional): The dataset split, either "train" (default) or "val"
        transforms (callable, optional): A function/transform that takes in
            ``img1, img2, flow, valid_flow_mask`` and returns a transformed version.
            ``valid_flow_mask`` is expected for consistency with other datasets which
            return a built-in valid mask, such as :class:`~torchvision.datasets.KittiFlow`.
    """

    def __init__(self, root: str, split: str='train', transforms: Optional[Callable]=None) -> None:
        super().__init__(root=root, transforms=transforms)
        verify_str_arg(split, 'split', valid_values=('train', 'val'))
        root = Path(root) / 'FlyingChairs'
        images = sorted(glob(str(root / 'data' / '*.ppm')))
        flows = sorted(glob(str(root / 'data' / '*.flo')))
        split_file_name = 'FlyingChairs_train_val.txt'
        if not os.path.exists(root / split_file_name):
            raise FileNotFoundError('The FlyingChairs_train_val.txt file was not found - please download it from the dataset page (see docstring).')
        split_list = np.loadtxt(str(root / split_file_name), dtype=np.int32)
        for i in range(len(flows)):
            split_id = split_list[i]
            if split == 'train' and split_id == 1 or (split == 'val' and split_id == 2):
                self._flow_list += [flows[i]]
                self._image_list += [[images[2 * i], images[2 * i + 1]]]

    def __getitem__(self, index: int) -> Union[T1, T2]:
        """Return example at given index.

        Args:
            index(int): The index of the example to retrieve

        Returns:
            tuple: A 3-tuple with ``(img1, img2, flow)``.
            The flow is a numpy array of shape (2, H, W) and the images are PIL images.
            ``flow`` is None if ``split="val"``.
            If a valid flow mask is generated within the ``transforms`` parameter,
            a 4-tuple with ``(img1, img2, flow, valid_flow_mask)`` is returned.
        """
        return super().__getitem__(index)

    def _read_flow(self, file_name: str) -> np.ndarray:
        return _read_flo(file_name)