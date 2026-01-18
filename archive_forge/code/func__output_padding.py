import math
import warnings
import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from .. import functional as F
from .. import init
from .lazy import LazyModuleMixin
from .module import Module
from .utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch._torch_docs import reproducibility_notes
from ..common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union
def _output_padding(self, input: Tensor, output_size: Optional[List[int]], stride: List[int], padding: List[int], kernel_size: List[int], num_spatial_dims: int, dilation: Optional[List[int]]=None) -> List[int]:
    if output_size is None:
        ret = _single(self.output_padding)
    else:
        has_batch_dim = input.dim() == num_spatial_dims + 2
        num_non_spatial_dims = 2 if has_batch_dim else 1
        if len(output_size) == num_non_spatial_dims + num_spatial_dims:
            output_size = output_size[num_non_spatial_dims:]
        if len(output_size) != num_spatial_dims:
            raise ValueError('ConvTranspose{}D: for {}D input, output_size must have {} or {} elements (got {})'.format(num_spatial_dims, input.dim(), num_spatial_dims, num_non_spatial_dims + num_spatial_dims, len(output_size)))
        min_sizes = torch.jit.annotate(List[int], [])
        max_sizes = torch.jit.annotate(List[int], [])
        for d in range(num_spatial_dims):
            dim_size = (input.size(d + num_non_spatial_dims) - 1) * stride[d] - 2 * padding[d] + (dilation[d] if dilation is not None else 1) * (kernel_size[d] - 1) + 1
            min_sizes.append(dim_size)
            max_sizes.append(min_sizes[d] + stride[d] - 1)
        for i in range(len(output_size)):
            size = output_size[i]
            min_size = min_sizes[i]
            max_size = max_sizes[i]
            if size < min_size or size > max_size:
                raise ValueError(f'requested an output size of {output_size}, but valid sizes range from {min_sizes} to {max_sizes} (for an input of {input.size()[2:]})')
        res = torch.jit.annotate(List[int], [])
        for d in range(num_spatial_dims):
            res.append(output_size[d] - min_sizes[d])
        ret = res
    return ret