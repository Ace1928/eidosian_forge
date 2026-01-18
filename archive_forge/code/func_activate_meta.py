import math
from enum import Enum
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch._prims_common as utils
from torch import SymBool, SymFloat, Tensor
from torch._decomp import (
from torch._ops import OpOverload
from torch._prims import _prim_elementwise_meta, ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _broadcast_shapes, _maybe_broadcast
from torch.utils import _pytree as pytree
import torch._refs
import torch._refs.nn.functional
import torch._refs.special
def activate_meta():
    activate_meta_table = {}
    for type in ['meta', 'post_autograd', 'pre_autograd']:
        registry = global_decomposition_table[type]
        for opo in registry:
            if opo not in activate_meta_table:
                activate_meta_table[opo] = registry[opo]
    for op_overload, fn in activate_meta_table.items():
        if isinstance(op_overload, torch._ops.HigherOrderOperator):
            continue
        assert isinstance(op_overload, OpOverload)
        op_overload.py_impl(torch._C.DispatchKey.Meta)(fn)
        if torch._C._dispatch_has_kernel_for_dispatch_key(op_overload.name(), 'CompositeImplicitAutograd'):
            if op_overload in global_decomposition_table['meta']:
                raise RuntimeError(f"{op_overload} is a CompositeImplicitAutograd op, we shouldn't register meta function for it. Instead, we should let the decomposition run and write meta kernels for the base operators.")
            pass
        elif op_overload.is_view:
            pass
        elif op_overload.name() in {'aten::empty_strided', 'aten::clone', 'aten::_to_copy', 'aten::copy_', 'aten::constant_pad_nd', 'aten::rot90', 'aten::as_strided_scatter'}:
            pass
        elif 'mkldnn::' in op_overload.name():
            _meta_lib_dont_use_me_use_register_meta_for_mkldnn.impl(op_overload, fn)
        elif 'mkl::' in op_overload.name():
            _meta_lib_dont_use_me_use_register_meta_for_mkl.impl(op_overload, fn)
        elif 'onednn::' in op_overload.name():
            _meta_lib_dont_use_me_use_register_meta_for_onednn.impl(op_overload, fn)
        elif 'quantized::' in op_overload.name():
            _meta_lib_dont_use_me_use_register_meta_for_quantized.impl(op_overload, fn)
        else:
            _meta_lib_dont_use_me_use_register_meta.impl(op_overload, fn)