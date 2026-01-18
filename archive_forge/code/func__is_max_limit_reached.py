import inspect
from contextlib import contextmanager
from typing import Any, Callable, ContextManager, Generator, Optional, Tuple, Type
import torch
import torch.distributed as dist
from torch import Tensor
import pytorch_lightning as pl
from lightning_fabric.utilities.distributed import _distributed_is_initialized
from lightning_fabric.utilities.imports import _TORCH_EQUAL_2_0
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.accelerators.xla import XLAAccelerator
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.loops import _Loop
from pytorch_lightning.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher, _PrefetchDataFetcher
from pytorch_lightning.loops.progress import _BaseProgress
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.strategies.parallel import ParallelStrategy
from pytorch_lightning.strategies.strategy import Strategy
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
def _is_max_limit_reached(current: int, maximum: int=-1) -> bool:
    """Check if the limit has been reached (if enabled).

    Args:
        current: the current value
        maximum: the maximum value (or -1 to disable limit)

    Returns:
        bool: whether the limit has been reached

    """
    return maximum != -1 and current >= maximum