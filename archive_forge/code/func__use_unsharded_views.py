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
@no_type_check
def _use_unsharded_views(self, as_params: bool) -> None:
    """
        Unflattens the unsharded flat parameter by setting the original
        parameter variables to be views into it.

        Args:
            as_params (bool): If ``True``, then registers the original
                parameters as ``nn.Parameter`` s; if ``False``, then registers
                the original parameters only as ``Tensor`` s. ``False`` should
                be used during forward/backward computation and when hiding the
                original parameters from :meth:`nn.Module.named_parameters`.
        """
    flat_param = self.flat_param
    self._check_unsharded(flat_param)
    views = self._get_unflat_views()
    from torch.distributed._tensor import DTensor
    for i, (view, (param_name, module, _)) in enumerate(zip(views, flat_param._param_infos)):
        if self._use_orig_params and as_params:
            if type(view) is DTensor:
                self._setattr_param(module, param_name, nn.Parameter(view))
                continue
            param = self.flat_param._params[i]
            self._setattr_param(module, param_name, param)
            param.data = view
        elif as_params:
            self._setattr_param(module, param_name, nn.Parameter(view))
        else:
            param_var: Tensor = view
            if self._use_orig_params:
                if self._training_state == HandleTrainingState.FORWARD:
                    self.flat_param._tensors[i] = view
                elif self._training_state == HandleTrainingState.BACKWARD_PRE:
                    tensor = self.flat_param._tensors[i]
                    tensor.data = view
                    param_var = tensor
            self._setattr_tensor(module, param_name, param_var)
            if self._use_orig_params and self._training_state == HandleTrainingState.FORWARD:
                module._parameters[param_name] = param_var
    for i, (param_name, module, _, prim_param_name, prim_module, _) in enumerate(self.flat_param._shared_param_infos):
        prim_param: Union[Tensor, nn.Parameter] = getattr(prim_module, prim_param_name)
        _p_assert(not as_params or isinstance(prim_param, nn.Parameter), f'as_params={as_params} type(prim_param)={type(prim_param)}')
        if self._use_orig_params and as_params:
            shared_param = self.flat_param._shared_params[i]
            self._setattr_param(module, param_name, shared_param)
            shared_param.data = prim_param
        elif as_params:
            self._setattr_param(module, param_name, prim_param)
        else:
            self._setattr_tensor(module, param_name, prim_param)
            if self._use_orig_params and self._training_state == HandleTrainingState.FORWARD:
                module._parameters[param_name] = prim_param