import warnings
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Type, Union
import PIL.Image
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import transforms as _transforms, tv_tensors
from torchvision.transforms.v2 import functional as F, Transform
from ._utils import _parse_labels_getter, _setup_number_or_seq, _setup_size, get_bounding_boxes, has_any, is_pure_tensor
class ConvertImageDtype(Transform):
    """[BETA] [DEPRECATED] Use ``v2.ToDtype(dtype, scale=True)`` instead.

    Convert input image to the given ``dtype`` and scale the values accordingly.

    .. v2betastatus:: ConvertImageDtype transform

    .. warning::
        Consider using ``ToDtype(dtype, scale=True)`` instead. See :class:`~torchvision.transforms.v2.ToDtype`.

    This function does not support PIL Image.

    Args:
        dtype (torch.dtype): Desired data type of the output

    .. note::

        When converting from a smaller to a larger integer ``dtype`` the maximum values are **not** mapped exactly.
        If converted back and forth, this mismatch has no effect.

    Raises:
        RuntimeError: When trying to cast :class:`torch.float32` to :class:`torch.int32` or :class:`torch.int64` as
            well as for trying to cast :class:`torch.float64` to :class:`torch.int64`. These conversions might lead to
            overflow errors since the floating point ``dtype`` cannot store consecutive integers over the whole range
            of the integer ``dtype``.
    """
    _v1_transform_cls = _transforms.ConvertImageDtype

    def __init__(self, dtype: torch.dtype=torch.float32) -> None:
        super().__init__()
        self.dtype = dtype

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.to_dtype, inpt, dtype=self.dtype, scale=True)