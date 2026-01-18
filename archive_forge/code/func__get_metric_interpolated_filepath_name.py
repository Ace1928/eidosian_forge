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
def _get_metric_interpolated_filepath_name(self, monitor_candidates: Dict[str, Tensor], trainer: 'pl.Trainer', del_filepath: Optional[str]=None) -> str:
    filepath = self.format_checkpoint_name(monitor_candidates)
    if self._enable_version_counter:
        version_cnt = self.STARTING_VERSION
        while self.file_exists(filepath, trainer) and filepath != del_filepath:
            filepath = self.format_checkpoint_name(monitor_candidates, ver=version_cnt)
            version_cnt += 1
    return filepath