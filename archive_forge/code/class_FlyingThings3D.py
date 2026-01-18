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
class FlyingThings3D(FlowDataset):
    """`FlyingThings3D <https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html>`_ dataset for optical flow.

    The dataset is expected to have the following structure: ::

        root
            FlyingThings3D
                frames_cleanpass
                    TEST
                    TRAIN
                frames_finalpass
                    TEST
                    TRAIN
                optical_flow
                    TEST
                    TRAIN

    Args:
        root (string): Root directory of the intel FlyingThings3D Dataset.
        split (string, optional): The dataset split, either "train" (default) or "test"
        pass_name (string, optional): The pass to use, either "clean" (default) or "final" or "both". See link above for
            details on the different passes.
        camera (string, optional): Which camera to return images from. Can be either "left" (default) or "right" or "both".
        transforms (callable, optional): A function/transform that takes in
            ``img1, img2, flow, valid_flow_mask`` and returns a transformed version.
            ``valid_flow_mask`` is expected for consistency with other datasets which
            return a built-in valid mask, such as :class:`~torchvision.datasets.KittiFlow`.
    """

    def __init__(self, root: str, split: str='train', pass_name: str='clean', camera: str='left', transforms: Optional[Callable]=None) -> None:
        super().__init__(root=root, transforms=transforms)
        verify_str_arg(split, 'split', valid_values=('train', 'test'))
        split = split.upper()
        verify_str_arg(pass_name, 'pass_name', valid_values=('clean', 'final', 'both'))
        passes = {'clean': ['frames_cleanpass'], 'final': ['frames_finalpass'], 'both': ['frames_cleanpass', 'frames_finalpass']}[pass_name]
        verify_str_arg(camera, 'camera', valid_values=('left', 'right', 'both'))
        cameras = ['left', 'right'] if camera == 'both' else [camera]
        root = Path(root) / 'FlyingThings3D'
        directions = ('into_future', 'into_past')
        for pass_name, camera, direction in itertools.product(passes, cameras, directions):
            image_dirs = sorted(glob(str(root / pass_name / split / '*/*')))
            image_dirs = sorted((Path(image_dir) / camera for image_dir in image_dirs))
            flow_dirs = sorted(glob(str(root / 'optical_flow' / split / '*/*')))
            flow_dirs = sorted((Path(flow_dir) / direction / camera for flow_dir in flow_dirs))
            if not image_dirs or not flow_dirs:
                raise FileNotFoundError('Could not find the FlyingThings3D flow images. Please make sure the directory structure is correct.')
            for image_dir, flow_dir in zip(image_dirs, flow_dirs):
                images = sorted(glob(str(image_dir / '*.png')))
                flows = sorted(glob(str(flow_dir / '*.pfm')))
                for i in range(len(flows) - 1):
                    if direction == 'into_future':
                        self._image_list += [[images[i], images[i + 1]]]
                        self._flow_list += [flows[i]]
                    elif direction == 'into_past':
                        self._image_list += [[images[i + 1], images[i]]]
                        self._flow_list += [flows[i + 1]]

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
        return _read_pfm(file_name)