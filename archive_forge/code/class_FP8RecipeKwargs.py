import argparse
import copy
import enum
import functools
import os
import typing
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, get_args
import torch
from .constants import FSDP_AUTO_WRAP_POLICY, FSDP_BACKWARD_PREFETCH, FSDP_SHARDING_STRATEGY, FSDP_STATE_DICT_TYPE
from .environment import str_to_bool
from .imports import is_cuda_available, is_npu_available, is_xpu_available
from .versions import compare_versions
@dataclass
class FP8RecipeKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize the initialization of the recipe for FP8 mixed precision
    training with `transformer-engine` or `ms-amp`.

    <Tip>

        For more information on `transformer-engine` args, please refer to the API
        [documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html).

        For more information on the `ms-amp` args, please refer to the Optimization Level
        [documentation](https://azure.github.io/MS-AMP/docs/user-tutorial/optimization-level).

    </Tip>

    ```python
    from accelerate import Accelerator
    from accelerate.utils import FP8RecipeKwargs

    kwargs = FP8RecipeKwargs(backend="te", fp8_format="HYBRID")
    accelerator = Accelerator(mixed_precision="fp8", kwargs_handlers=[kwargs])
    ```

    To use MS-AMP as an engine, pass `backend="msamp"` and the `optimization_level`:

    ```python
    kwargs = FP8RecipeKwargs(backend="msamp", optimization_level="02")
    ```

    Args:
        backend (`str`, *optional*, defaults to "msamp"):
            Which FP8 engine to use. Must be one of `"msamp"` (MS-AMP) or `"te"` (TransformerEngine).
        margin (`int`, *optional*, default to 0):
            The margin to use for the gradient scaling.
        interval (`int`, *optional*, default to 1):
            The interval to use for how often the scaling factor is recomputed.
        fp8_format (`str`, *optional*, default to "E4M3"):
            The format to use for the FP8 recipe. Must be one of `E4M3` or `HYBRID`.
        amax_history_len (`int`, *optional*, default to 1024):
            The length of the history to use for the scaling factor computation
        amax_compute_algo (`str`, *optional*, default to "most_recent"):
            The algorithm to use for the scaling factor computation. Must be one of `max` or `most_recent`.
        override_linear_precision (`tuple` of three `bool`, *optional*, default to `(False, False, False)`):
            Whether or not to execute `fprop`, `dgrad`, and `wgrad` GEMMS in higher precision.
        optimization_level (`str`), one of `O1`, `O2`. (default is `O2`):
            What level of 8-bit collective communication should be used with MS-AMP. In general:
                * O1: Weight gradients and `all_reduce` communications are done in fp8, reducing GPU
                    memory usage and communication bandwidth
                * O2: First-order optimizer states are in 8-bit, and second order states are in FP16.
                    Only available when using Adam or AdamW. This maintains accuracy and can potentially save the
                    highest memory.
                * 03: Specifically for DeepSpeed, implements capabilities so weights and master weights of models
                    are stored in FP8. If `fp8` is selected and deepspeed is enabled, will be used by default. (Not
                    available currently).
    """
    backend: Backend = 'MSAMP'
    opt_level: OptLevel = 'O2'
    margin: int = 0
    interval: int = 1
    fp8_format: FP8Format = 'E4M3'
    amax_history_len: int = 1
    amax_compute_algo: AmaxComputeAlgorithm = 'most_recent'
    override_linear_precision: Tuple[bool, bool, bool] = (False, False, False)

    def __post_init__(self):
        if self.backend.upper() not in get_args(Backend):
            raise ValueError("`backend` must be 'MSAMP' or 'TE' (TransformerEngine).")
        self.backend = self.backend.upper()
        if self.backend == 'TE':
            self.fp8_format = self.fp8_format.upper()
            if self.fp8_format not in get_args(FP8Format):
                raise ValueError(f'`fp8_format` must be one of {' or '.join(get_args(FP8Format))}.')
            if self.amax_compute_algo not in get_args(AmaxComputeAlgorithm):
                raise ValueError(f'`amax_compute_algo` must be one of {' or '.join(get_args(AmaxComputeAlgorithm))}')
        elif self.backend == 'MSAMP':
            if self.opt_level not in get_args(OptLevel):
                raise ValueError(f'`optimization_level` must be one of {' or '.join(get_args(OptLevel))}')