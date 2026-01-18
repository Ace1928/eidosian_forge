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
def _save_monitor_checkpoint(self, trainer: 'pl.Trainer', monitor_candidates: Dict[str, Tensor]) -> None:
    assert self.monitor
    current = monitor_candidates.get(self.monitor)
    if self.check_monitor_top_k(trainer, current):
        assert current is not None
        self._update_best_and_save(current, trainer, monitor_candidates)
    elif self.verbose:
        epoch = monitor_candidates['epoch']
        step = monitor_candidates['step']
        rank_zero_info(f'Epoch {epoch:d}, global step {step:d}: {self.monitor!r} was not in top {self.save_top_k}')