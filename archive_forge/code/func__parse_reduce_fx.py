from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union, cast
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torchmetrics import Metric
from typing_extensions import TypedDict, override
from lightning_fabric.utilities import move_data_to_device
from lightning_fabric.utilities.apply_func import convert_tensors_to_scalars
from lightning_fabric.utilities.distributed import _distributed_is_initialized
from lightning_fabric.utilities.imports import _TORCH_EQUAL_2_0, _TORCH_GREATER_EQUAL_2_0
from pytorch_lightning.utilities.data import extract_batch_size
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _TORCHMETRICS_GREATER_EQUAL_1_0_0
from pytorch_lightning.utilities.memory import recursive_detach
from pytorch_lightning.utilities.rank_zero import WarningCache, rank_zero_warn
from pytorch_lightning.utilities.warnings import PossibleUserWarning
def _parse_reduce_fx(self) -> None:
    error = f'Only `self.log(..., reduce_fx={{min,max,mean,sum}})` are supported. If you need a custom reduction, please log a `torchmetrics.Metric` instance instead. Found: {self.reduce_fx}'
    if isinstance(self.reduce_fx, str):
        reduce_fx = self.reduce_fx.lower()
        if reduce_fx == 'avg':
            reduce_fx = 'mean'
        if reduce_fx not in ('min', 'max', 'mean', 'sum'):
            raise MisconfigurationException(error)
        self.reduce_fx = getattr(torch, reduce_fx)
    elif self.is_custom_reduction:
        raise MisconfigurationException(error)