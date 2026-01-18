import math
import numbers
import warnings
from typing import Any, Callable, cast, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union
import PIL.Image
import torch
from torchvision import transforms as _transforms, tv_tensors
from torchvision.ops.boxes import box_iou
from torchvision.transforms.functional import _get_perspective_coeffs
from torchvision.transforms.v2 import functional as F, InterpolationMode, Transform
from torchvision.transforms.v2.functional._geometry import _check_interpolation
from torchvision.transforms.v2.functional._utils import _FillType
from ._transform import _RandomApplyTransform
from ._utils import (
class RandomIoUCrop(Transform):
    """[BETA] Random IoU crop transformation from
    `"SSD: Single Shot MultiBox Detector" <https://arxiv.org/abs/1512.02325>`_.

    .. v2betastatus:: RandomIoUCrop transform

    This transformation requires an image or video data and ``tv_tensors.BoundingBoxes`` in the input.

    .. warning::
        In order to properly remove the bounding boxes below the IoU threshold, `RandomIoUCrop`
        must be followed by :class:`~torchvision.transforms.v2.SanitizeBoundingBoxes`, either immediately
        after or later in the transforms pipeline.

    If the input is a :class:`torch.Tensor` or a ``TVTensor`` (e.g. :class:`~torchvision.tv_tensors.Image`,
    :class:`~torchvision.tv_tensors.Video`, :class:`~torchvision.tv_tensors.BoundingBoxes` etc.)
    it can have arbitrary number of leading batch dimensions. For example,
    the image can have ``[..., C, H, W]`` shape. A bounding box can have ``[..., 4]`` shape.

    Args:
        min_scale (float, optional): Minimum factors to scale the input size.
        max_scale (float, optional): Maximum factors to scale the input size.
        min_aspect_ratio (float, optional): Minimum aspect ratio for the cropped image or video.
        max_aspect_ratio (float, optional): Maximum aspect ratio for the cropped image or video.
        sampler_options (list of float, optional): List of minimal IoU (Jaccard) overlap between all the boxes and
            a cropped image or video. Default, ``None`` which corresponds to ``[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]``
        trials (int, optional): Number of trials to find a crop for a given value of minimal IoU (Jaccard) overlap.
            Default, 40.
    """

    def __init__(self, min_scale: float=0.3, max_scale: float=1.0, min_aspect_ratio: float=0.5, max_aspect_ratio: float=2.0, sampler_options: Optional[List[float]]=None, trials: int=40):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        if sampler_options is None:
            sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.options = sampler_options
        self.trials = trials

    def _check_inputs(self, flat_inputs: List[Any]) -> None:
        if not (has_all(flat_inputs, tv_tensors.BoundingBoxes) and has_any(flat_inputs, PIL.Image.Image, tv_tensors.Image, is_pure_tensor)):
            raise TypeError(f'{type(self).__name__}() requires input sample to contain tensor or PIL images and bounding boxes. Sample can also contain masks.')

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        orig_h, orig_w = query_size(flat_inputs)
        bboxes = get_bounding_boxes(flat_inputs)
        while True:
            idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]
            if min_jaccard_overlap >= 1.0:
                return dict()
            for _ in range(self.trials):
                r = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(2)
                new_w = int(orig_w * r[0])
                new_h = int(orig_h * r[1])
                aspect_ratio = new_w / new_h
                if not self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                    continue
                r = torch.rand(2)
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue
                xyxy_bboxes = F.convert_bounding_box_format(bboxes.as_subclass(torch.Tensor), bboxes.format, tv_tensors.BoundingBoxFormat.XYXY)
                cx = 0.5 * (xyxy_bboxes[..., 0] + xyxy_bboxes[..., 2])
                cy = 0.5 * (xyxy_bboxes[..., 1] + xyxy_bboxes[..., 3])
                is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                if not is_within_crop_area.any():
                    continue
                xyxy_bboxes = xyxy_bboxes[is_within_crop_area]
                ious = box_iou(xyxy_bboxes, torch.tensor([[left, top, right, bottom]], dtype=xyxy_bboxes.dtype, device=xyxy_bboxes.device))
                if ious.max() < min_jaccard_overlap:
                    continue
                return dict(top=top, left=left, height=new_h, width=new_w, is_within_crop_area=is_within_crop_area)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if len(params) < 1:
            return inpt
        output = self._call_kernel(F.crop, inpt, top=params['top'], left=params['left'], height=params['height'], width=params['width'])
        if isinstance(output, tv_tensors.BoundingBoxes):
            output[~params['is_within_crop_area']] = 0
        return output