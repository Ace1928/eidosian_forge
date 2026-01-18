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
class _ResultCollection(dict):
    """Collection (dictionary) of :class:`~pytorch_lightning.trainer.connectors.logger_connector.result._ResultMetric`

    Example::

        # you can log to a specific collection.
        # arguments: fx, key, value, metadata

        result = _ResultCollection(training=True)
        result.log('training_step', 'acc', torch.tensor(...), on_step=True, on_epoch=True)
        result.log('validation_step', 'recall', torch.tensor(...), on_step=True, on_epoch=True)

    """
    DATALOADER_SUFFIX = '/dataloader_idx_{}'

    def __init__(self, training: bool) -> None:
        super().__init__()
        self.training = training
        self.batch: Optional[Any] = None
        self.batch_size: Optional[int] = None
        self.dataloader_idx: Optional[int] = None

    @property
    def result_metrics(self) -> List[_ResultMetric]:
        return list(self.values())

    def _extract_batch_size(self, value: _ResultMetric, batch_size: Optional[int], meta: _Metadata) -> int:
        if batch_size is None:
            batch_size = self.batch_size
        if batch_size is not None:
            return batch_size
        batch_size = 1
        if self.batch is not None and value.is_tensor and meta.on_epoch and meta.is_mean_reduction:
            batch_size = extract_batch_size(self.batch)
            self.batch_size = batch_size
        return batch_size

    def log(self, fx: str, name: str, value: _VALUE, prog_bar: bool=False, logger: bool=True, on_step: bool=False, on_epoch: bool=True, reduce_fx: Callable='mean' if _TORCH_EQUAL_2_0 else torch.mean, enable_graph: bool=False, sync_dist: bool=False, sync_dist_fn: Callable=_Sync.no_op, sync_dist_group: Optional[Any]=None, add_dataloader_idx: bool=True, batch_size: Optional[int]=None, metric_attribute: Optional[str]=None, rank_zero_only: bool=False) -> None:
        """See :meth:`~pytorch_lightning.core.LightningModule.log`"""
        if not enable_graph:
            value = recursive_detach(value)
        key = f'{fx}.{name}'
        if add_dataloader_idx and self.dataloader_idx is not None:
            key += f'.{self.dataloader_idx}'
            fx += f'.{self.dataloader_idx}'
        meta = _Metadata(fx=fx, name=name, prog_bar=prog_bar, logger=logger, on_step=on_step, on_epoch=on_epoch, reduce_fx=reduce_fx, enable_graph=enable_graph, add_dataloader_idx=add_dataloader_idx, dataloader_idx=self.dataloader_idx, metric_attribute=metric_attribute)
        meta.sync = _Sync(_should=sync_dist, fn=sync_dist_fn, _group=sync_dist_group, rank_zero_only=rank_zero_only)
        if key not in self:
            self.register_key(key, meta, value)
        elif meta != self[key].meta:
            raise MisconfigurationException(f'You called `self.log({name}, ...)` twice in `{fx}` with different arguments. This is not allowed')
        batch_size = self._extract_batch_size(self[key], batch_size, meta)
        self.update_metrics(key, value, batch_size)

    def register_key(self, key: str, meta: _Metadata, value: _VALUE) -> None:
        """Create one _ResultMetric object per value.

        Value can be provided as a nested collection

        """
        metric = _ResultMetric(meta, isinstance(value, Tensor)).to(value.device)
        self[key] = metric

    def update_metrics(self, key: str, value: _VALUE, batch_size: int) -> None:
        result_metric = self[key]
        result_metric.forward(value, batch_size)
        result_metric.has_reset = False

    @staticmethod
    def _get_cache(result_metric: _ResultMetric, on_step: bool) -> Optional[Tensor]:
        cache = None
        if on_step and result_metric.meta.on_step:
            cache = result_metric._forward_cache
        elif not on_step and result_metric.meta.on_epoch:
            if result_metric._computed is None:
                should = result_metric.meta.sync.should
                if not should and result_metric.is_tensor and _distributed_is_initialized():
                    warning_cache.warn(f'It is recommended to use `self.log({result_metric.meta.name!r}, ..., sync_dist=True)` when logging on epoch level in distributed setting to accumulate the metric across devices.', category=PossibleUserWarning)
                result_metric.compute()
                result_metric.meta.sync.should = should
            cache = result_metric._computed
        if cache is not None:
            if not isinstance(cache, Tensor):
                raise ValueError(f'The `.compute()` return of the metric logged as {result_metric.meta.name!r} must be a tensor. Found {cache}')
            if not result_metric.meta.enable_graph:
                return cache.detach()
        return cache

    def valid_items(self) -> Generator:
        """This function is used to iterate over current valid metrics."""
        return ((k, v) for k, v in self.items() if not v.has_reset and self.dataloader_idx == v.meta.dataloader_idx)

    def _forked_name(self, result_metric: _ResultMetric, on_step: bool) -> Tuple[str, str]:
        name = result_metric.meta.name
        forked_name = result_metric.meta.forked_name(on_step)
        add_dataloader_idx = result_metric.meta.add_dataloader_idx
        dl_idx = result_metric.meta.dataloader_idx
        if add_dataloader_idx and dl_idx is not None:
            dataloader_suffix = self.DATALOADER_SUFFIX.format(dl_idx)
            name += dataloader_suffix
            forked_name += dataloader_suffix
        return (name, forked_name)

    def metrics(self, on_step: bool) -> _METRICS:
        metrics = _METRICS(callback={}, log={}, pbar={})
        for _, result_metric in self.valid_items():
            value = self._get_cache(result_metric, on_step)
            if not isinstance(value, Tensor):
                continue
            name, forked_name = self._forked_name(result_metric, on_step)
            if result_metric.meta.logger:
                metrics['log'][forked_name] = value
            if self.training or (result_metric.meta.on_epoch and (not on_step)):
                metrics['callback'][name] = value
                metrics['callback'][forked_name] = value
            if result_metric.meta.prog_bar:
                metrics['pbar'][forked_name] = convert_tensors_to_scalars(value)
        return metrics

    def reset(self, metrics: Optional[bool]=None, fx: Optional[str]=None) -> None:
        """Reset the result collection.

        Args:
            metrics: If True, only ``torchmetrics.Metric`` results are reset,
                if False, only ``torch.Tensors`` are reset,
                if ``None``, both are.
            fx: Function to reset

        """
        for item in self.values():
            requested_type = metrics is None or metrics ^ item.is_tensor
            same_fx = fx is None or fx == item.meta.fx
            if requested_type and same_fx:
                item.reset()

    def to(self, *args: Any, **kwargs: Any) -> '_ResultCollection':
        """Move all data to the given device."""
        self.update(apply_to_collection(dict(self), (Tensor, Metric), move_data_to_device, *args, **kwargs))
        return self

    def cpu(self) -> '_ResultCollection':
        """Move all data to CPU."""
        return self.to(device='cpu')

    def __str__(self) -> str:
        self_str = str({k: v for k, v in self.items() if v})
        return f'{self.__class__.__name__}({self_str})'

    def __repr__(self) -> str:
        return f'{{{self.training}, {super().__repr__()}}}'