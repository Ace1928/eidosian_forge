import logging
import os
import uuid
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple
import pytorch_lightning as pl
from pytorch_lightning.utilities.memory import garbage_collection_cuda, is_oom_error
from pytorch_lightning.utilities.parsing import lightning_getattr, lightning_setattr
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
def _reset_progress(trainer: 'pl.Trainer') -> None:
    if trainer.lightning_module.automatic_optimization:
        trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.reset()
    else:
        trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.reset()
    trainer.fit_loop.epoch_progress.reset()