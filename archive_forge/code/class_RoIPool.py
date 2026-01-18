from typing import List, Union
import torch
import torch.fx
from torch import nn, Tensor
from torch.jit.annotations import BroadcastingList2
from torch.nn.modules.utils import _pair
from torchvision.extension import _assert_has_ops
from ..utils import _log_api_usage_once
from ._utils import check_roi_boxes_shape, convert_boxes_to_roi_format
class RoIPool(nn.Module):
    """
    See :func:`roi_pool`.
    """

    def __init__(self, output_size: BroadcastingList2[int], spatial_scale: float):
        super().__init__()
        _log_api_usage_once(self)
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input: Tensor, rois: Union[Tensor, List[Tensor]]) -> Tensor:
        return roi_pool(input, rois, self.output_size, self.spatial_scale)

    def __repr__(self) -> str:
        s = f'{self.__class__.__name__}(output_size={self.output_size}, spatial_scale={self.spatial_scale})'
        return s