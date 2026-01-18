import functools
from contextlib import nullcontext
from typing import Any, Callable, Dict, Optional, Sequence
import torch
import torch._decomp
import torch._prims
import torch._refs
import torch._refs.nn
import torch._refs.nn.functional
import torch._refs.special
import torch.overrides
from torch._prims_common import torch_function_passthrough
class TorchRefsMode(torch.overrides.TorchFunctionMode):
    """
    Switches the interpretation of torch.* functions and Tensor methods to
    use PrimTorch refs in torch._refs.  (Direct calls to _refs are unaffected.)

    >>> # xdoctest: +SKIP
    >>> with TorchRefsMode():
    ...     torch.add(x, y)  # calls torch._refs.add(x, y)

    By default, this context manager will fall back on the torch.* if the
    ref does not exist; set strict=True to error if this occurs.
    If the ref exists we still would like to fall back on the torch.* sometimes,
    this behavior can be customized by passing a function to should_fallback_fn.
    """

    def __init__(self, strict=False, should_fallback_fn=lambda *_: False, prims_mode_cls=nullcontext):
        self.strict = strict
        self.should_fallback_fn = should_fallback_fn
        self.prims_mode_cls = prims_mode_cls

    def __torch_function__(self, orig_func: Callable, types: Sequence, args: Sequence[Any]=(), kwargs: Optional[Dict]=None):
        if kwargs is None:
            kwargs = {}
        if orig_func in torch_function_passthrough or orig_func in all_prims():
            with self.prims_mode_cls():
                return orig_func(*args, **kwargs)
        mapping = torch_to_refs_map()
        func = mapping.get(orig_func, None)
        if func is None and isinstance(orig_func, torch._ops.OpOverload):
            func = torch._decomp.decomposition_table.get(orig_func, None)
        if func is not None:
            if self.should_fallback_fn(self, orig_func, func, args, kwargs):
                return orig_func(*args, **kwargs)
            with self:
                return func(*args, **kwargs)
        if self.strict:
            raise RuntimeError(f'no _refs support for {torch.overrides.resolve_name(orig_func)}')
        return orig_func(*args, **kwargs)