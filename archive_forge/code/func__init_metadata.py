import contextlib
import functools
import logging
import os
import warnings
from enum import auto, Enum
from itertools import accumulate, chain
from typing import (
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.fsdp._common_utils import (
from torch.distributed.utils import _alloc_storage, _free_storage, _p_assert
from torch.nn.parameter import _ParameterMeta  # type: ignore[attr-defined]
from ._fsdp_extensions import _ext_post_unflatten_transform, _ext_pre_flatten_transform
@classmethod
def _init_metadata(cls, self, param_infos: List[ParamInfo], numels: List[int], shapes: List[torch.Size], fqns: List[str], shared_param_infos: List[SharedParamInfo], param_extensions: List[Optional[Any]], params: Optional[List[nn.Parameter]], shared_params: Optional[List[nn.Parameter]], is_padding_mask: List[bool]) -> None:
    """
        Initializes attributes holding metadata about the original parameters
        comprising the flat parameter.

        We expose this method separate from the constructor to keep the
        constructor only responsible for the flat parameter's tensor data. This
        method should only be called once per model, while the constructor may
        be called multiple times, e.g. when reloading from a checkpoint, in
        which case only the tensor data needs to be passed to the constructor.
        Since :meth:`load_state_dict` is implemented via :meth:`copy_`, the
        metadata is correctly assumed to be unchanged.

        Args:
            See the Attributes in the class docstring.
        """
    assert len(param_infos) == len(shapes)
    assert len(param_infos) == len(fqns)
    assert len(param_infos) == len(param_extensions)
    self._num_params = len(param_infos)
    self._param_infos = param_infos
    self._shapes = shapes
    self._fqns = fqns
    self._param_extensions = param_extensions
    self._is_padding_mask = is_padding_mask
    numels_without_padding: List[int] = []
    for numel, is_padding in zip(numels, is_padding_mask):
        if not is_padding:
            numels_without_padding.append(numel)
    self._numels = tuple(numels_without_padding)
    self._numels_with_padding = tuple(numels)
    assert len(self._numels) == self._num_params
    self._shared_param_infos = tuple(shared_param_infos)
    self._modules = {pi.module for pi in self._param_infos}.union({spi.module for spi in self._shared_param_infos})
    assert (params is None) == (shared_params is None)
    if params is not None:
        assert shared_params is not None and len(shared_params) == len(shared_param_infos)
        self._params = []
        for param, is_padding in zip(params, is_padding_mask):
            if not is_padding:
                self._params.append(param)
        self._shared_params = shared_params
        for param in chain(self._params, self._shared_params):
            _set_fsdp_flattened(param)
        self._is_grad_none_mask = [False for _ in range(self._num_params)]
        self._tensors = [None for _ in range(self._num_params)]
    else:
        self._params = None
        self._shared_params = None
        self._is_grad_none_mask = None
        self._tensors = None
    self._unpadded_unsharded_size = self.size()
    _set_fsdp_flattened(self)
    self._post_backward_called = False