import itertools
import os
from dataclasses import dataclass
from multiprocessing.queues import SimpleQueue
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional
import torch
import torch.backends.cudnn
import torch.multiprocessing as mp
from lightning_utilities import apply_to_collection
from torch.nn import Module
from typing_extensions import override
from lightning_fabric.accelerators.cpu import CPUAccelerator
from lightning_fabric.strategies.launchers.launcher import _Launcher
from lightning_fabric.utilities.apply_func import move_data_to_device
from lightning_fabric.utilities.distributed import _set_num_threads_if_needed
from lightning_fabric.utilities.imports import _IS_INTERACTIVE
from lightning_fabric.utilities.seed import _collect_rng_states, _set_rng_states
def _check_bad_cuda_fork() -> None:
    """Checks whether it is safe to fork and initialize CUDA in the new processes, and raises an exception if not.

    The error message replaces PyTorch's 'Cannot re-initialize CUDA in forked subprocess' with helpful advice for
    Lightning users.

    """
    if not torch.cuda.is_initialized():
        return
    message = "Lightning can't create new processes if CUDA is already initialized. Did you manually call `torch.cuda.*` functions, have moved the model to the device, or allocated memory on the GPU any other way? Please remove any such calls, or change the selected strategy."
    if _IS_INTERACTIVE:
        message += ' You will have to restart the Python kernel.'
    raise RuntimeError(message)