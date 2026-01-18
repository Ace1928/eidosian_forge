from typing import Collection, Dict, Iterator, List, Optional, Set, Tuple, cast
import torch
from torch import Tensor
from torchmetrics.utilities import rank_zero_warn
def _prepocess_inputs(things: Set[int], stuffs: Set[int], inputs: Tensor, void_color: Tuple[int, int], allow_unknown_category: bool) -> Tensor:
    """Preprocesses an input tensor for metric calculation.

    NOTE: The input tensor is assumed to have dimension ordering (B, spatial_dim0, ..., spatial_dim_N, 2).
    Spelled out explicitly, this means (B, num_points, 2) for point clouds, (B, H, W, 2) for images, and so on.

    Args:
        things: All category IDs for things classes.
        stuffs: All category IDs for stuff classes.
        inputs: The input tensor.
        void_color: An additional color that is masked out during metrics calculation.
        allow_unknown_category: If true, unknown category IDs are mapped to "void".
            Otherwise, an exception is raised if they occur.

    Returns:
        The preprocessed input tensor flattened along the spatial dimensions.

    """
    out = inputs.detach().clone()
    out = torch.flatten(out, 1, -2)
    mask_stuffs = _isin(out[:, :, 0], list(stuffs))
    mask_things = _isin(out[:, :, 0], list(things))
    mask_stuffs_instance = torch.stack([torch.zeros_like(mask_stuffs), mask_stuffs], dim=-1)
    out[mask_stuffs_instance] = 0
    if not allow_unknown_category and (not torch.all(mask_things | mask_stuffs)):
        raise ValueError(f'Unknown categories found: {out[~(mask_things | mask_stuffs)]}')
    out[~(mask_things | mask_stuffs)] = out.new(void_color)
    return out