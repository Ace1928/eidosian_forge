from abc import ABC, abstractmethod
import contextlib
from typing import Any
import torch
import torch.utils._pytree as pytree
from torch._C._functorch import (
from torch.autograd.forward_ad import _set_fwd_grad_enabled
class FunctionalizeInterpreter(FuncTorchInterpreter):

    def __init__(self, cdata: CInterpreter):
        assert cdata.key() == TransformType.Functionalize
        self._cdata = cdata
        self._cptr = CFunctionalizeInterpreterPtr(cdata)

    def process(self, op, args, kwargs):
        kernel = op.functorch_table[TransformType.Functionalize]
        return kernel(self, *args, **kwargs)

    def functionalize_add_back_views(self):
        return self._cptr.functionalizeAddBackViews()