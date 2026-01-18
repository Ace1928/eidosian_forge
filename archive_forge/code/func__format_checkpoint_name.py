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
def _format_checkpoint_name(self, filename: Optional[str], metrics: Dict[str, Tensor], prefix: str='', auto_insert_metric_name: bool=True) -> str:
    if not filename:
        filename = '{epoch}' + self.CHECKPOINT_JOIN_CHAR + '{step}'
    groups = re.findall('(\\{.*?)[:\\}]', filename)
    groups = sorted(groups, key=lambda x: len(x), reverse=True)
    for group in groups:
        name = group[1:]
        if auto_insert_metric_name:
            filename = filename.replace(group, name + self.CHECKPOINT_EQUALS_CHAR + '{' + name)
        filename = filename.replace(group, f'{{0[{name}]')
        if name not in metrics:
            metrics[name] = torch.tensor(0)
    filename = filename.format(metrics)
    if prefix:
        filename = self.CHECKPOINT_JOIN_CHAR.join([prefix, filename])
    return filename