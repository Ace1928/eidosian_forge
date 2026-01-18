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
def _should_save_on_train_epoch_end(self, trainer: 'pl.Trainer') -> bool:
    if self._save_on_train_epoch_end is not None:
        return self._save_on_train_epoch_end
    if trainer.check_val_every_n_epoch != 1:
        return False
    num_val_batches = sum(trainer.num_val_batches) if isinstance(trainer.num_val_batches, list) else trainer.num_val_batches
    if num_val_batches == 0:
        return True
    return trainer.val_check_interval == 1.0