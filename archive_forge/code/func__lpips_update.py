import inspect
import os
from typing import List, NamedTuple, Optional, Tuple, Union
import torch
from torch import Tensor, nn
from typing_extensions import Literal
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE, _TORCHVISION_GREATER_EQUAL_0_13
def _lpips_update(img1: Tensor, img2: Tensor, net: nn.Module, normalize: bool) -> Tuple[Tensor, Union[int, Tensor]]:
    if not (_valid_img(img1, normalize) and _valid_img(img2, normalize)):
        raise ValueError(f'Expected both input arguments to be normalized tensors with shape [N, 3, H, W]. Got input with shape {img1.shape} and {img2.shape} and values in range {[img1.min(), img1.max()]} and {[img2.min(), img2.max()]} when all values are expected to be in the {([0, 1] if normalize else [-1, 1])} range.')
    loss = net(img1, img2, normalize=normalize).squeeze()
    return (loss, img1.shape[0])