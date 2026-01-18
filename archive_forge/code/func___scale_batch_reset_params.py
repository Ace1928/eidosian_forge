import logging
import os
import uuid
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple
import pytorch_lightning as pl
from pytorch_lightning.utilities.memory import garbage_collection_cuda, is_oom_error
from pytorch_lightning.utilities.parsing import lightning_getattr, lightning_setattr
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
def __scale_batch_reset_params(trainer: 'pl.Trainer', steps_per_trial: int) -> None:
    from pytorch_lightning.loggers.logger import DummyLogger
    trainer.logger = DummyLogger() if trainer.logger is not None else None
    trainer.callbacks = []
    loop = trainer._active_loop
    assert loop is not None
    if isinstance(loop, pl.loops._FitLoop):
        trainer.limit_train_batches = 1.0
        trainer.limit_val_batches = steps_per_trial
        trainer.fit_loop.epoch_loop.max_steps = steps_per_trial
    elif isinstance(loop, pl.loops._EvaluationLoop):
        stage = trainer.state.stage
        assert stage is not None
        setattr(trainer, f'limit_{stage.dataloader_prefix}_batches', steps_per_trial)
        loop.verbose = False