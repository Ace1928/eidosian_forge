import warnings
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Type, Union
import PIL.Image
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import transforms as _transforms, tv_tensors
from torchvision.transforms.v2 import functional as F, Transform
from ._utils import _parse_labels_getter, _setup_number_or_seq, _setup_size, get_bounding_boxes, has_any, is_pure_tensor
class SanitizeBoundingBoxes(Transform):
    """[BETA] Remove degenerate/invalid bounding boxes and their corresponding labels and masks.

    .. v2betastatus:: SanitizeBoundingBoxes transform

    This transform removes bounding boxes and their associated labels/masks that:

    - are below a given ``min_size``: by default this also removes degenerate boxes that have e.g. X2 <= X1.
    - have any coordinate outside of their corresponding image. You may want to
      call :class:`~torchvision.transforms.v2.ClampBoundingBoxes` first to avoid undesired removals.

    It is recommended to call it at the end of a pipeline, before passing the
    input to the models. It is critical to call this transform if
    :class:`~torchvision.transforms.v2.RandomIoUCrop` was called.
    If you want to be extra careful, you may call it after all transforms that
    may modify bounding boxes but once at the end should be enough in most
    cases.

    Args:
        min_size (float, optional) The size below which bounding boxes are removed. Default is 1.
        labels_getter (callable or str or None, optional): indicates how to identify the labels in the input.
            By default, this will try to find a "labels" key in the input (case-insensitive), if
            the input is a dict or it is a tuple whose second element is a dict.
            This heuristic should work well with a lot of datasets, including the built-in torchvision datasets.
            It can also be a callable that takes the same input
            as the transform, and returns the labels.
    """

    def __init__(self, min_size: float=1.0, labels_getter: Union[Callable[[Any], Optional[torch.Tensor]], str, None]='default') -> None:
        super().__init__()
        if min_size < 1:
            raise ValueError(f'min_size must be >= 1, got {min_size}.')
        self.min_size = min_size
        self.labels_getter = labels_getter
        self._labels_getter = _parse_labels_getter(labels_getter)

    def forward(self, *inputs: Any) -> Any:
        inputs = inputs if len(inputs) > 1 else inputs[0]
        labels = self._labels_getter(inputs)
        if labels is not None and (not isinstance(labels, torch.Tensor)):
            raise ValueError(f'The labels in the input to forward() must be a tensor or None, got {type(labels)} instead.')
        flat_inputs, spec = tree_flatten(inputs)
        boxes = get_bounding_boxes(flat_inputs)
        if labels is not None and boxes.shape[0] != labels.shape[0]:
            raise ValueError(f'Number of boxes (shape={boxes.shape}) and number of labels (shape={labels.shape}) do not match.')
        boxes = cast(tv_tensors.BoundingBoxes, F.convert_bounding_box_format(boxes, new_format=tv_tensors.BoundingBoxFormat.XYXY))
        ws, hs = (boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
        valid = (ws >= self.min_size) & (hs >= self.min_size) & (boxes >= 0).all(dim=-1)
        image_h, image_w = boxes.canvas_size
        valid &= (boxes[:, 0] <= image_w) & (boxes[:, 2] <= image_w)
        valid &= (boxes[:, 1] <= image_h) & (boxes[:, 3] <= image_h)
        params = dict(valid=valid.as_subclass(torch.Tensor), labels=labels)
        flat_outputs = [self._transform(inpt, params) for inpt in flat_inputs]
        return tree_unflatten(flat_outputs, spec)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        is_label = inpt is not None and inpt is params['labels']
        is_bounding_boxes_or_mask = isinstance(inpt, (tv_tensors.BoundingBoxes, tv_tensors.Mask))
        if not (is_label or is_bounding_boxes_or_mask):
            return inpt
        output = inpt[params['valid']]
        if is_label:
            return output
        return tv_tensors.wrap(output, like=inpt)