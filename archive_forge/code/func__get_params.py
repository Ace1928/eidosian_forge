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
def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
    lam = float(self._dist.sample(()))
    H, W = query_size(flat_inputs)
    r_x = torch.randint(W, size=(1,))
    r_y = torch.randint(H, size=(1,))
    r = 0.5 * math.sqrt(1.0 - lam)
    r_w_half = int(r * W)
    r_h_half = int(r * H)
    x1 = int(torch.clamp(r_x - r_w_half, min=0))
    y1 = int(torch.clamp(r_y - r_h_half, min=0))
    x2 = int(torch.clamp(r_x + r_w_half, max=W))
    y2 = int(torch.clamp(r_y + r_h_half, max=H))
    box = (x1, y1, x2, y2)
    lam_adjusted = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))
    return dict(box=box, lam_adjusted=lam_adjusted)