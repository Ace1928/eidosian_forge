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
class Sintel(FlowDataset):
    """`Sintel <http://sintel.is.tue.mpg.de/>`_ Dataset for optical flow.

    The dataset is expected to have the following structure: ::

        root
            Sintel
                testing
                    clean
                        scene_1
                        scene_2
                        ...
                    final
                        scene_1
                        scene_2
                        ...
                training
                    clean
                        scene_1
                        scene_2
                        ...
                    final
                        scene_1
                        scene_2
                        ...
                    flow
                        scene_1
                        scene_2
                        ...

    Args:
        root (string): Root directory of the Sintel Dataset.
        split (string, optional): The dataset split, either "train" (default) or "test"
        pass_name (string, optional): The pass to use, either "clean" (default), "final", or "both". See link above for
            details on the different passes.
        transforms (callable, optional): A function/transform that takes in
            ``img1, img2, flow, valid_flow_mask`` and returns a transformed version.
            ``valid_flow_mask`` is expected for consistency with other datasets which
            return a built-in valid mask, such as :class:`~torchvision.datasets.KittiFlow`.
    """

    def __init__(self, root: str, split: str='train', pass_name: str='clean', transforms: Optional[Callable]=None) -> None:
        super().__init__(root=root, transforms=transforms)
        verify_str_arg(split, 'split', valid_values=('train', 'test'))
        verify_str_arg(pass_name, 'pass_name', valid_values=('clean', 'final', 'both'))
        passes = ['clean', 'final'] if pass_name == 'both' else [pass_name]
        root = Path(root) / 'Sintel'
        flow_root = root / 'training' / 'flow'
        for pass_name in passes:
            split_dir = 'training' if split == 'train' else split
            image_root = root / split_dir / pass_name
            for scene in os.listdir(image_root):
                image_list = sorted(glob(str(image_root / scene / '*.png')))
                for i in range(len(image_list) - 1):
                    self._image_list += [[image_list[i], image_list[i + 1]]]
                if split == 'train':
                    self._flow_list += sorted(glob(str(flow_root / scene / '*.flo')))

    def __getitem__(self, index: int) -> Union[T1, T2]:
        """Return example at given index.

        Args:
            index(int): The index of the example to retrieve

        Returns:
            tuple: A 3-tuple with ``(img1, img2, flow)``.
            The flow is a numpy array of shape (2, H, W) and the images are PIL images.
            ``flow`` is None if ``split="test"``.
            If a valid flow mask is generated within the ``transforms`` parameter,
            a 4-tuple with ``(img1, img2, flow, valid_flow_mask)`` is returned.
        """
        return super().__getitem__(index)

    def _read_flow(self, file_name: str) -> np.ndarray:
        return _read_flo(file_name)