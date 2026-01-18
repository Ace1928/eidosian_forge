import inspect
from copy import deepcopy
from functools import wraps
from typing import (
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torch import nn as nn
from torch.nn.modules.module import _IncompatibleKeys
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing_extensions import override
from lightning_fabric.plugins import Precision
from lightning_fabric.strategies import Strategy
from lightning_fabric.utilities import move_data_to_device
from lightning_fabric.utilities.data import _set_sampler_epoch
from lightning_fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning_fabric.utilities.types import Optimizable
class _FabricDataLoader:

    def __init__(self, dataloader: DataLoader, device: Optional[torch.device]=None) -> None:
        """The FabricDataLoader is a wrapper for the :class:`~torch.utils.data.DataLoader`. It moves the data to the
        device automatically if the device is specified.

        Args:
            dataloader: The dataloader to wrap
            device: The device to which the data should be moved. By default the device is `None` and no data
                transfers will be made (identical behavior as :class:`~torch.utils.data.DataLoader`).

        """
        self.__dict__.update(dataloader.__dict__)
        self._dataloader = dataloader
        self._device = device
        self._num_iter_calls = 0

    @property
    def device(self) -> Optional[torch.device]:
        return self._device

    def __len__(self) -> int:
        return len(self._dataloader)

    def __iter__(self) -> Union[Iterator[Any], Generator[Any, None, None]]:
        _set_sampler_epoch(self._dataloader, self._num_iter_calls)
        self._num_iter_calls += 1
        if self._device is None:
            yield from iter(self._dataloader)
        else:
            for item in self._dataloader:
                yield move_data_to_device(item, self._device)