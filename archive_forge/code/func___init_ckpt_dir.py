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
def __init_ckpt_dir(self, dirpath: Optional[_PATH], filename: Optional[str]) -> None:
    self._fs = get_filesystem(dirpath if dirpath else '')
    if dirpath and _is_local_file_protocol(dirpath if dirpath else ''):
        dirpath = os.path.realpath(os.path.expanduser(dirpath))
    self.dirpath = dirpath
    self.filename = filename