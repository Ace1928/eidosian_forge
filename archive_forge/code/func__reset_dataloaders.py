import logging
import os
import uuid
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple
import pytorch_lightning as pl
from pytorch_lightning.utilities.memory import garbage_collection_cuda, is_oom_error
from pytorch_lightning.utilities.parsing import lightning_getattr, lightning_setattr
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
def _reset_dataloaders(trainer: 'pl.Trainer') -> None:
    loop = trainer._active_loop
    assert loop is not None
    loop._combined_loader = None
    loop.setup_data()
    if isinstance(loop, pl.loops._FitLoop):
        loop.epoch_loop.val_loop._combined_loader = None
        loop.epoch_loop.val_loop.setup_data()