import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Tuple
import torch
def _combine_masks(method, t, mask):
    """Combine the masks from all pruning methods and returns a new mask.

            Args:
                method (a BasePruningMethod subclass): pruning method
                    currently being applied.
                t (torch.Tensor): tensor representing the parameter to prune
                    (of same dimensions as mask).
                mask (torch.Tensor): mask from previous pruning iteration

            Returns:
                new_mask (torch.Tensor): new mask that combines the effects
                    of the old mask and the new mask from the current
                    pruning method (of same dimensions as mask and t).
            """
    new_mask = mask
    new_mask = new_mask.to(dtype=t.dtype)
    if method.PRUNING_TYPE == 'unstructured':
        slc = mask == 1
    elif method.PRUNING_TYPE == 'structured':
        if not hasattr(method, 'dim'):
            raise AttributeError('Pruning methods of PRUNING_TYPE "structured" need to have the attribute `dim` defined.')
        n_dims = t.dim()
        dim = method.dim
        if dim < 0:
            dim = n_dims + dim
        if dim < 0:
            raise IndexError(f'Index is out of bounds for tensor with dimensions {n_dims}')
        keep_channel = mask.sum(dim=[d for d in range(n_dims) if d != dim]) != 0
        slc = [slice(None)] * n_dims
        slc[dim] = keep_channel
    elif method.PRUNING_TYPE == 'global':
        n_dims = len(t.shape)
        slc = [slice(None)] * n_dims
    else:
        raise ValueError(f'Unrecognized PRUNING_TYPE {method.PRUNING_TYPE}')
    partial_mask = method.compute_mask(t[slc], default_mask=mask[slc])
    new_mask[slc] = partial_mask.to(dtype=new_mask.dtype)
    return new_mask