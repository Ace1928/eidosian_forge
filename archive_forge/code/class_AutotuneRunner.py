import itertools
import random
from typing import Tuple
from .. import language as tl
from ..language import core as lcore
from . import torch_wrapper
from .core import ExecutionContext
from .memory_map import MemoryMap
from .tl_lang import (TritonLangProxy, WrappedTensor, _primitive_to_tensor,
class AutotuneRunner:

    def __init__(self, func, autotune_params, grid=None):
        self.func = func
        self.autotune_params = autotune_params
        self.grid = grid

    def __call__(self, *args, **kwargs):
        assert len(self.autotune_params['configs']) >= 1
        for config in self.autotune_params['configs'][1:]:

            def convert_arg(v):
                if torch.is_tensor(v):
                    return torch.clone(v)
                return v
            new_args = tuple(map(convert_arg, args))
            new_kwargs = {k: convert_arg(v) for k, v in kwargs.items()}
            if self.grid:
                self.func[self.grid](*new_args, **new_kwargs, **config.kwargs)
            else:
                self.func(*new_args, **new_kwargs, **config.kwargs)
        main_config = self.autotune_params['configs'][0]
        if self.grid:
            self.func[self.grid](*args, **kwargs, **main_config.kwargs)
        else:
            self.func(*args, **kwargs, **main_config.kwargs)