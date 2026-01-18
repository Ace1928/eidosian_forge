import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
class GridExecutor:

    def __init__(self, fn, arg_names, grid):
        from .jit import _normalize_ty
        self.fn = fn
        self.arg_names = arg_names
        self.grid = grid
        __annotations__ = {name: _normalize_ty(ty) for name, ty in fn.__annotations__.items()}
        self.constexprs = [name for name in arg_names if __annotations__.get(name) == 'constexpr']

    def _patch_lang(self, builder):
        lang = [value for _, value in self.fn.__globals__.items() if value in [tl, tl.core]]
        assert len(lang) == 1, "triton.language must be visible from within jit'd function"
        _patch_lang_tensor(getattr(lang[0], 'tensor'), builder)
        _patch_lang_core(lang[0], builder)
        _patch_lang_math(lang[0], builder)

    def __call__(self, *args_dev, **kwargs):
        args_hst = [_unwrap(arg).cpu() if hasattr(arg, 'data_ptr') else arg for arg in args_dev]
        kwargs = {k: v for k, v in kwargs.items() if k not in RESERVED_KWS}
        self._patch_lang(builder)
        args = inspect.getcallargs(self.fn, *args_hst, **kwargs)
        args = {name: arg if name in self.constexprs else _implicit_cvt(arg) for name, arg in args.items()}
        grid = self.grid(args) if callable(self.grid) else self.grid
        assert len(grid) <= 3
        grid = grid + (1,) * (3 - len(grid))
        builder.set_grid_dim(*grid)
        for x in range(grid[0]):
            for y in range(grid[1]):
                for z in range(grid[2]):
                    builder.set_grid_idx(x, y, z)
                    self.fn(**args)
        for arg_dev, arg_hst in zip(args_dev, args_hst):
            if hasattr(arg_dev, 'data_ptr'):
                _unwrap(arg_dev).copy_(arg_hst.to(arg_dev.device))