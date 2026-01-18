from abc import ABC, abstractmethod
import contextlib
from typing import Any
import torch
import torch.utils._pytree as pytree
from torch._C._functorch import (
from torch.autograd.forward_ad import _set_fwd_grad_enabled
class JvpInterpreter(FuncTorchInterpreter):

    def __init__(self, cdata: CInterpreter):
        assert cdata.key() == TransformType.Jvp
        self._cdata = cdata
        self._cptr = CJvpInterpreterPtr(cdata)

    def lift(self, args, kwargs):
        args, kwargs = pytree.tree_map_only(torch.Tensor, self._cptr.lift, [args, kwargs])
        return (args, kwargs)

    def process(self, op, args, kwargs):
        kernel = op.functorch_table[TransformType.Jvp]
        args, kwargs = self.lift(args, kwargs)
        return kernel(self, *args, **kwargs)

    def lower(self):
        prev_fwd_grad_mode = self.prev_fwd_grad_mode()
        if not prev_fwd_grad_mode:
            return nested(_set_fwd_grad_enabled(False), super().lower())
        return super().lower()

    def prev_fwd_grad_mode(self):
        return self._cptr.prevFwdGradMode()