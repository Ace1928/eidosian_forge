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
def __resolve_ckpt_dir(self, trainer: 'pl.Trainer') -> _PATH:
    """Determines model checkpoint save directory at runtime. Reference attributes from the trainer's logger to
        determine where to save checkpoints. The path for saving weights is set in this priority:

        1.  The ``ModelCheckpoint``'s ``dirpath`` if passed in
        2.  The ``Logger``'s ``log_dir`` if the trainer has loggers
        3.  The ``Trainer``'s ``default_root_dir`` if the trainer has no loggers

        The path gets extended with subdirectory "checkpoints".

        """
    if self.dirpath is not None:
        return self.dirpath
    if len(trainer.loggers) > 0:
        if trainer.loggers[0].save_dir is not None:
            save_dir = trainer.loggers[0].save_dir
        else:
            save_dir = trainer.default_root_dir
        name = trainer.loggers[0].name
        version = trainer.loggers[0].version
        version = version if isinstance(version, str) else f'version_{version}'
        ckpt_path = os.path.join(save_dir, str(name), version, 'checkpoints')
    else:
        ckpt_path = os.path.join(trainer.default_root_dir, 'checkpoints')
    return ckpt_path