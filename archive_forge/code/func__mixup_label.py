import math
import numbers
import warnings
from typing import Any, Callable, Dict, List, Tuple
import PIL.Image
import torch
from torch.nn.functional import one_hot
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import transforms as _transforms, tv_tensors
from torchvision.transforms.v2 import functional as F
from ._transform import _RandomApplyTransform, Transform
from ._utils import _parse_labels_getter, has_any, is_pure_tensor, query_chw, query_size
def _mixup_label(self, label: torch.Tensor, *, lam: float) -> torch.Tensor:
    label = one_hot(label, num_classes=self.num_classes)
    if not label.dtype.is_floating_point:
        label = label.float()
    return label.roll(1, 0).mul_(1.0 - lam).add_(label.mul(lam))