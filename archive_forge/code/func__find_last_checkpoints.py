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
def _find_last_checkpoints(self, trainer: 'pl.Trainer') -> Set[str]:
    ckpt_path = self.__resolve_ckpt_dir(trainer)
    last_pattern = f'^{self.CHECKPOINT_NAME_LAST}(-(\\d+))?'

    def _is_last(path: Path) -> bool:
        return path.suffix == self.FILE_EXTENSION and bool(re.match(last_pattern, path.stem))
    if self._fs.exists(ckpt_path):
        return {os.path.normpath(p) for p in self._fs.ls(ckpt_path, detail=False) if _is_last(Path(p))}
    return set()