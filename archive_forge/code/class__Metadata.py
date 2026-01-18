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
@dataclass
class _Metadata:
    fx: str
    name: str
    prog_bar: bool = False
    logger: bool = True
    on_step: bool = False
    on_epoch: bool = True
    reduce_fx: Callable = 'mean' if _TORCH_EQUAL_2_0 else torch.mean
    enable_graph: bool = False
    add_dataloader_idx: bool = True
    dataloader_idx: Optional[int] = None
    metric_attribute: Optional[str] = None
    _sync: Optional[_Sync] = None

    def __post_init__(self) -> None:
        if not self.on_step and (not self.on_epoch):
            raise MisconfigurationException('`self.log(on_step=False, on_epoch=False)` is not useful.')
        self._parse_reduce_fx()

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

    @property
    def sync(self) -> _Sync:
        assert self._sync is not None
        return self._sync

    @sync.setter
    def sync(self, sync: _Sync) -> None:
        if sync.op is None:
            sync.op = self.reduce_fx.__name__
        self._sync = sync

    @property
    def forked(self) -> bool:
        return self.on_step and self.on_epoch

    def forked_name(self, on_step: bool) -> str:
        if self.forked:
            return f'{self.name}_{('step' if on_step else 'epoch')}'
        return self.name

    @property
    def is_mean_reduction(self) -> bool:
        return self.reduce_fx is torch.mean

    @property
    def is_sum_reduction(self) -> bool:
        return self.reduce_fx in (torch.sum, sum)

    @property
    def is_max_reduction(self) -> bool:
        return self.reduce_fx in (torch.max, max)

    @property
    def is_min_reduction(self) -> bool:
        return self.reduce_fx in (torch.min, min)

    @property
    def is_custom_reduction(self) -> bool:
        return not (self.is_mean_reduction or self.is_max_reduction or self.is_min_reduction or self.is_sum_reduction)