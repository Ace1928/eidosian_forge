import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
class InterpretedFunction:

    def _patch_lang(self, builder):
        lang = [value for _, value in self.fn.__globals__.items() if value in [tl, tl.core]]
        assert len(lang) == 1, "triton.language must be visible from within jit'd function"
        _patch_lang_tensor(getattr(lang[0], 'tensor'), builder)
        _patch_lang_core(lang[0], builder)

    def __init__(self, fn) -> None:
        self.fn = fn

        def run(*args, **kwargs):
            grid = kwargs['grid']
            kwargs = {k: v for k, v in kwargs.items() if k not in RESERVED_KWS + ['grid']}
            return GridExecutor(self.fn, self.arg_names, grid)(*args, **kwargs)
        self.run = run
        signature = inspect.signature(fn)
        self.arg_names = [v.name for v in signature.parameters.values()]

    def __getitem__(self, grid):
        return GridExecutor(self.fn, self.arg_names, grid)

    def __call__(self, *args, **kwargs):
        self._patch_lang(builder)
        return self.fn(*args, **kwargs)