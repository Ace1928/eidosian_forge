from __future__ import annotations
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple, Union
import torch
from torch.utils._pytree import tree_flatten
from ._tv_tensor import TVTensor
[BETA] :class:`torch.Tensor` subclass for bounding boxes.

    .. note::
        There should be only one :class:`~torchvision.tv_tensors.BoundingBoxes`
        instance per sample e.g. ``{"img": img, "bbox": BoundingBoxes(...)}``,
        although one :class:`~torchvision.tv_tensors.BoundingBoxes` object can
        contain multiple bounding boxes.

    Args:
        data: Any data that can be turned into a tensor with :func:`torch.as_tensor`.
        format (BoundingBoxFormat, str): Format of the bounding box.
        canvas_size (two-tuple of ints): Height and width of the corresponding image or video.
        dtype (torch.dtype, optional): Desired data type of the bounding box. If omitted, will be inferred from
            ``data``.
        device (torch.device, optional): Desired device of the bounding box. If omitted and ``data`` is a
            :class:`torch.Tensor`, the device is taken from it. Otherwise, the bounding box is constructed on the CPU.
        requires_grad (bool, optional): Whether autograd should record operations on the bounding box. If omitted and
            ``data`` is a :class:`torch.Tensor`, the value is taken from it. Otherwise, defaults to ``False``.
    