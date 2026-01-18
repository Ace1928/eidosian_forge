import itertools
import random
from typing import Tuple
from .. import language as tl
from ..language import core as lcore
from . import torch_wrapper
from .core import ExecutionContext
from .memory_map import MemoryMap
from .tl_lang import (TritonLangProxy, WrappedTensor, _primitive_to_tensor,
class DebuggerFunction:

    def __init__(self, func, grid=(1,)):
        self.func = func
        self.grid = grid

    def _is_constexpr(self, name):
        return name in self.func.__annotations__ and self.func.__annotations__[name] is lcore.constexpr

    def _get_constexpr(self):
        result = []
        for name, annotation in self.func.__annotations__.items():
            if annotation is lcore.constexpr:
                result.append(name)
        return result

    def _assert_constexpr(self, **kwargs):
        constexp = self._get_constexpr()
        missing = [i for i in constexp if i not in kwargs.keys()]
        assert len(missing) == 0, f'You must specify constexpr {missing}'

    def _get_grid(self, **kwargs):
        if callable(self.grid):
            return self.grid(kwargs)
        else:
            return self.grid

    def __call__(self, *args, **kwargs):
        self._assert_constexpr(**kwargs)
        memory = MemoryMap()

        def convert_arg(v):
            name, arg = v
            if torch.is_tensor(arg):
                ptr = memory.add_tensor(arg)
                return WrappedTensor(torch.tensor([ptr], dtype=torch.int64, device='cuda'))
            if self._is_constexpr(name):
                return debugger_constexpr(arg)
            return WrappedTensor(_primitive_to_tensor(arg))
        new_args = tuple(map(convert_arg, zip(self.func.__code__.co_varnames, args)))
        new_kwargs = {k: convert_arg((k, v)) for k, v in kwargs.items() if k not in ['num_warps', 'num_stages']}
        grid = self._get_grid(**kwargs)
        for program_id in program_ids_from_grid(grid):
            proxy = TritonLangProxy(memory, ExecutionContext(program_id, grid))
            attach_triton(tl, proxy)
            self.func(*new_args, **new_kwargs)
            detach_triton(tl)