import logging
import os
import re
import shutil
import time
import warnings
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Set
from weakref import proxy
import torch
import yaml
from torch import Tensor
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.cloud_io import _is_dir, _is_local_file_protocol, get_filesystem
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.callbacks import Checkpoint
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import WarningCache, rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.types import STEP_OUTPUT
def _update_best_and_save(self, current: Tensor, trainer: 'pl.Trainer', monitor_candidates: Dict[str, Tensor]) -> None:
    k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k
    del_filepath = None
    if len(self.best_k_models) == k and k > 0:
        del_filepath = self.kth_best_model_path
        self.best_k_models.pop(del_filepath)
    if isinstance(current, Tensor) and torch.isnan(current):
        current = torch.tensor(float('inf' if self.mode == 'min' else '-inf'), device=current.device)
    filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer, del_filepath)
    self.current_score = current
    self.best_k_models[filepath] = current
    if len(self.best_k_models) == k:
        _op = max if self.mode == 'min' else min
        self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)
        self.kth_value = self.best_k_models[self.kth_best_model_path]
    _op = min if self.mode == 'min' else max
    self.best_model_path = _op(self.best_k_models, key=self.best_k_models.get)
    self.best_model_score = self.best_k_models[self.best_model_path]
    if self.verbose:
        epoch = monitor_candidates['epoch']
        step = monitor_candidates['step']
        rank_zero_info(f'Epoch {epoch:d}, global step {step:d}: {self.monitor!r} reached {current:0.5f} (best {self.best_model_score:0.5f}), saving model to {filepath!r} as top {k}')
    self._save_checkpoint(trainer, filepath)
    if del_filepath and self._should_remove_checkpoint(trainer, del_filepath, filepath):
        self._remove_checkpoint(trainer, del_filepath)