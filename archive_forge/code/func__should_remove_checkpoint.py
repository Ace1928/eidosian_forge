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
def _should_remove_checkpoint(self, trainer: 'pl.Trainer', previous: str, current: str) -> bool:
    """Checks if the previous checkpoint should be deleted.

        A checkpoint won't be deleted if any of the cases apply:
        - The previous checkpoint is the same as the current checkpoint (means the old was already overwritten by new)
        - The previous checkpoint is not in the current checkpoint directory and the filesystem is local
        - The previous checkpoint is the checkpoint the Trainer resumed from and the filesystem is local

        """
    if previous == current:
        return False
    if not _is_local_file_protocol(previous):
        return True
    previous = Path(previous).absolute()
    resume_path = Path(trainer.ckpt_path).absolute() if trainer.ckpt_path is not None else None
    if resume_path is not None and previous == resume_path:
        return False
    assert self.dirpath is not None
    dirpath = Path(self.dirpath).absolute()
    return dirpath in previous.parents