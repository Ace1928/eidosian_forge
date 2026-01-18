import math
import numbers
import warnings
from typing import Any, List, Optional, Sequence, Tuple, Union
import PIL.Image
import torch
from torch.nn.functional import grid_sample, interpolate, pad as torch_pad
from torchvision import tv_tensors
from torchvision.transforms import _functional_pil as _FP
from torchvision.transforms._functional_tensor import _pad_symmetric
from torchvision.transforms.functional import (
from torchvision.utils import _log_api_usage_once
from ._meta import _get_size_image_pil, clamp_bounding_boxes, convert_bounding_box_format
from ._utils import _FillTypeJIT, _get_kernel, _register_five_ten_crop_kernel_internal, _register_kernel_internal
def _affine_bounding_boxes_with_expand(bounding_boxes: torch.Tensor, format: tv_tensors.BoundingBoxFormat, canvas_size: Tuple[int, int], angle: Union[int, float], translate: List[float], scale: float, shear: List[float], center: Optional[List[float]]=None, expand: bool=False) -> Tuple[torch.Tensor, Tuple[int, int]]:
    if bounding_boxes.numel() == 0:
        return (bounding_boxes, canvas_size)
    original_shape = bounding_boxes.shape
    original_dtype = bounding_boxes.dtype
    bounding_boxes = bounding_boxes.clone() if bounding_boxes.is_floating_point() else bounding_boxes.float()
    dtype = bounding_boxes.dtype
    device = bounding_boxes.device
    bounding_boxes = convert_bounding_box_format(bounding_boxes, old_format=format, new_format=tv_tensors.BoundingBoxFormat.XYXY, inplace=True).reshape(-1, 4)
    angle, translate, shear, center = _affine_parse_args(angle, translate, scale, shear, InterpolationMode.NEAREST, center)
    if center is None:
        height, width = canvas_size
        center = [width * 0.5, height * 0.5]
    affine_vector = _get_inverse_affine_matrix(center, angle, translate, scale, shear, inverted=False)
    transposed_affine_matrix = torch.tensor(affine_vector, dtype=dtype, device=device).reshape(2, 3).T
    points = bounding_boxes[:, [[0, 1], [2, 1], [2, 3], [0, 3]]].reshape(-1, 2)
    points = torch.cat([points, torch.ones(points.shape[0], 1, device=device, dtype=dtype)], dim=-1)
    transformed_points = torch.matmul(points, transposed_affine_matrix)
    transformed_points = transformed_points.reshape(-1, 4, 2)
    out_bbox_mins, out_bbox_maxs = torch.aminmax(transformed_points, dim=1)
    out_bboxes = torch.cat([out_bbox_mins, out_bbox_maxs], dim=1)
    if expand:
        height, width = canvas_size
        points = torch.tensor([[0.0, 0.0, 1.0], [0.0, float(height), 1.0], [float(width), float(height), 1.0], [float(width), 0.0, 1.0]], dtype=dtype, device=device)
        new_points = torch.matmul(points, transposed_affine_matrix)
        tr = torch.amin(new_points, dim=0, keepdim=True)
        out_bboxes.sub_(tr.repeat((1, 2)))
        affine_vector = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
        new_width, new_height = _compute_affine_output_size(affine_vector, width, height)
        canvas_size = (new_height, new_width)
    out_bboxes = clamp_bounding_boxes(out_bboxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=canvas_size)
    out_bboxes = convert_bounding_box_format(out_bboxes, old_format=tv_tensors.BoundingBoxFormat.XYXY, new_format=format, inplace=True).reshape(original_shape)
    out_bboxes = out_bboxes.to(original_dtype)
    return (out_bboxes, canvas_size)