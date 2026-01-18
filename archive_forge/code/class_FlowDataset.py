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
class FlowDataset(ABC, VisionDataset):
    _has_builtin_flow_mask = False

    def __init__(self, root: str, transforms: Optional[Callable]=None) -> None:
        super().__init__(root=root)
        self.transforms = transforms
        self._flow_list: List[str] = []
        self._image_list: List[List[str]] = []

    def _read_img(self, file_name: str) -> Image.Image:
        img = Image.open(file_name)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    @abstractmethod
    def _read_flow(self, file_name: str):
        pass

    def __getitem__(self, index: int) -> Union[T1, T2]:
        img1 = self._read_img(self._image_list[index][0])
        img2 = self._read_img(self._image_list[index][1])
        if self._flow_list:
            flow = self._read_flow(self._flow_list[index])
            if self._has_builtin_flow_mask:
                flow, valid_flow_mask = flow
            else:
                valid_flow_mask = None
        else:
            flow = valid_flow_mask = None
        if self.transforms is not None:
            img1, img2, flow, valid_flow_mask = self.transforms(img1, img2, flow, valid_flow_mask)
        if self._has_builtin_flow_mask or valid_flow_mask is not None:
            return (img1, img2, flow, valid_flow_mask)
        else:
            return (img1, img2, flow)

    def __len__(self) -> int:
        return len(self._image_list)

    def __rmul__(self, v: int) -> torch.utils.data.ConcatDataset:
        return torch.utils.data.ConcatDataset([self] * v)