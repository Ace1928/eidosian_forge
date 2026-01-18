from contextlib import contextmanager
from itertools import chain
import typing
from typing import (
import torch
from torch import Tensor
import torch.nn as nn
from fairscale.internal.state_dict import replace_by_prefix_
def _init_flatten_params(self, p_set: Set[Tuple[nn.Module, str]]) -> Tuple[List[nn.Parameter], List[Tuple[str, nn.Module, str]], List[Tuple[str, str, nn.Module, str, nn.Module, str]]]:
    """Build metadata for need-to-be-flatten parameters and returns a list
            contains the need-to-be-flatten parameters.

            This also returns param_infos and shared_param_infos, which
            will be attached to the flat parameter object.

        Args:
            p_set (set):
                A set of (module, param_name) for a set of params that needed
                to be flattened. There could be shared params in this set.
        """
    param_infos = []
    shared_param_memo: Dict[nn.Parameter, Tuple[str, nn.Module, str]] = {}
    shared_param_infos = []
    params = []
    fp32 = []
    fp16 = []
    for module_name, m in self.named_modules():
        for n, p in m.named_parameters(recurse=False):
            if p.dtype != torch.float16:
                fp32.append(module_name)
            else:
                fp16.append(module_name)
            if p is not None and (m, n) in p_set:
                if p in shared_param_memo:
                    mname, shared_m, shared_n = shared_param_memo[p]
                    shared_param_infos.append((module_name, mname, m, n, shared_m, shared_n))
                else:
                    shared_param_memo[p] = (module_name, m, n)
                    param_infos.append((module_name, m, n))
                    params.append(p)
    del shared_param_memo
    fp16_msg, fp32_msg = (','.join(fp16), ','.join(fp32))
    assert len(set((p.dtype for p in params))) == 1, f'expects all parameters to have same dtype: fp32: {fp32_msg} \n fp16: {fp16_msg} '
    assert len(set((p.requires_grad for p in params))) == 1, f'expects all parameters to have same requires_grad {p_set}'
    assert len(params) == len(set(params)), 'params list should not have dups'
    return (params, param_infos, shared_param_infos)