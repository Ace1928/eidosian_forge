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
def __init_triggers(self, every_n_train_steps: Optional[int], every_n_epochs: Optional[int], train_time_interval: Optional[timedelta]) -> None:
    if every_n_train_steps is None and every_n_epochs is None and (train_time_interval is None):
        every_n_epochs = 1
        every_n_train_steps = 0
        log.debug('Both every_n_train_steps and every_n_epochs are not set. Setting every_n_epochs=1')
    else:
        every_n_epochs = every_n_epochs or 0
        every_n_train_steps = every_n_train_steps or 0
    self._train_time_interval: Optional[timedelta] = train_time_interval
    self._every_n_epochs: int = every_n_epochs
    self._every_n_train_steps: int = every_n_train_steps