from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _take_torch(tensor, indices, axis=None, **_):
    """Torch implementation of np.take"""
    torch = _i('torch')
    if not isinstance(indices, torch.Tensor):
        indices = torch.as_tensor(indices)
    if axis is None:
        return tensor.flatten()[indices]
    if indices.ndim == 1:
        if (indices < 0).any():
            dim_length = tensor.size()[0] if axis is None else tensor.shape[axis]
            indices = torch.where(indices >= 0, indices, indices + dim_length)
        return torch.index_select(tensor, dim=axis, index=indices)
    if axis == -1:
        return tensor[..., indices]
    fancy_indices = [slice(None)] * axis + [indices]
    return tensor[fancy_indices]