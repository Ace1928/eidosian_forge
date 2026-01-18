import logging
import os
import re
from typing import Any, Dict, Optional
import torch
from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem
from torch import Tensor
import pytorch_lightning as pl
from lightning_fabric.plugins.environments.slurm import SLURMEnvironment
from lightning_fabric.utilities.cloud_io import _is_dir, get_filesystem
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.precision import MixedPrecision
from pytorch_lightning.trainer import call
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _OMEGACONF_AVAILABLE
from pytorch_lightning.utilities.migration import pl_legacy_patch
from pytorch_lightning.utilities.migration.utils import _pl_migrate_checkpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
def _parse_ckpt_path(self, state_fn: TrainerFn, ckpt_path: Optional[_PATH], model_provided: bool, model_connected: bool) -> Optional[_PATH]:
    """Converts the ``ckpt_path`` special values into an actual filepath, depending on the trainer
        configuration."""
    if ckpt_path is None and SLURMEnvironment.detect() and (self._hpc_resume_path is not None):
        ckpt_path = 'hpc'
    from pytorch_lightning.callbacks.on_exception_checkpoint import OnExceptionCheckpoint
    ft_checkpoints = [cb for cb in self.trainer.callbacks if isinstance(cb, OnExceptionCheckpoint)]
    fn = state_fn.value
    if ckpt_path is None and ft_checkpoints and (self.trainer.state.fn == TrainerFn.FITTING):
        ckpt_path = 'last'
        rank_zero_warn(f"`.{fn}(ckpt_path=None)` was called without a model. The last model of the previous `fit` call will be used. You can pass `{fn}(ckpt_path='best')` to use the best model or `{fn}(ckpt_path='last')` to use the last model. If you pass a value, this warning will be silenced.")
    if model_provided and ckpt_path is None:
        return None
    if model_connected and ckpt_path is None:
        ckpt_path = 'best'
        ft_tip = ' There is also an on-exception checkpoint available, however it is used by default only when fitting.' if ft_checkpoints else ''
        rank_zero_warn(f'`.{fn}(ckpt_path=None)` was called without a model. The best model of the previous `fit` call will be used.' + ft_tip + f" You can pass `.{fn}(ckpt_path='best')` to use the best model or `.{fn}(ckpt_path='last')` to use the last model. If you pass a value, this warning will be silenced.")
    if ckpt_path == 'best':
        if len(self.trainer.checkpoint_callbacks) > 1:
            rank_zero_warn(f'`.{fn}(ckpt_path="best")` is called with Trainer configured with multiple `ModelCheckpoint` callbacks. It will use the best checkpoint path from first checkpoint callback.')
        if not self.trainer.checkpoint_callback:
            raise ValueError(f'`.{fn}(ckpt_path="best")` is set but `ModelCheckpoint` is not configured.')
        has_best_model_path = self.trainer.checkpoint_callback.best_model_path
        if hasattr(self.trainer.checkpoint_callback, 'best_model_path') and (not has_best_model_path):
            if self.trainer.fast_dev_run:
                raise ValueError(f'You cannot execute `.{fn}(ckpt_path="best")` with `fast_dev_run=True`. Please pass an exact checkpoint path to `.{fn}(ckpt_path=...)`')
            raise ValueError(f'`.{fn}(ckpt_path="best")` is set but `ModelCheckpoint` is not configured to save the best model.')
        ckpt_path = getattr(self.trainer.checkpoint_callback, 'best_model_path', None)
    elif ckpt_path == 'last':
        candidates = {getattr(ft, 'ckpt_path', None) for ft in ft_checkpoints}
        for callback in self.trainer.checkpoint_callbacks:
            if isinstance(callback, ModelCheckpoint):
                candidates |= callback._find_last_checkpoints(self.trainer)
        candidates_fs = {path: get_filesystem(path) for path in candidates if path}
        candidates_ts = {path: fs.modified(path) for path, fs in candidates_fs.items() if fs.exists(path)}
        if not candidates_ts:
            rank_zero_warn(f'.{fn}(ckpt_path="last") is set, but there is no last checkpoint available. No checkpoint will be loaded. HINT: Set `ModelCheckpoint(..., save_last=True)`.')
            return None
        ckpt_path = max(candidates_ts, key=candidates_ts.get)
    elif ckpt_path == 'hpc':
        if not self._hpc_resume_path:
            raise ValueError(f'`.{fn}(ckpt_path="hpc")` is set but no HPC checkpoint was found. Please pass an exact checkpoint path to `.{{fn}}(ckpt_path=...)`')
        ckpt_path = self._hpc_resume_path
    if not ckpt_path:
        raise ValueError(f'`.{fn}()` found no path for the best weights: {ckpt_path!r}. Please specify a path for a checkpoint `.{fn}(ckpt_path=PATH)`')
    return ckpt_path