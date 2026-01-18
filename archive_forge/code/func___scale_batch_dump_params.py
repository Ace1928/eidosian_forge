import logging
import os
import uuid
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple
import pytorch_lightning as pl
from pytorch_lightning.utilities.memory import garbage_collection_cuda, is_oom_error
from pytorch_lightning.utilities.parsing import lightning_getattr, lightning_setattr
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
def __scale_batch_dump_params(trainer: 'pl.Trainer') -> Dict[str, Any]:
    dumped_params = {'loggers': trainer.loggers, 'callbacks': trainer.callbacks}
    loop = trainer._active_loop
    assert loop is not None
    if isinstance(loop, pl.loops._FitLoop):
        dumped_params['max_steps'] = trainer.max_steps
        dumped_params['limit_train_batches'] = trainer.limit_train_batches
        dumped_params['limit_val_batches'] = trainer.limit_val_batches
    elif isinstance(loop, pl.loops._EvaluationLoop):
        stage = trainer.state.stage
        assert stage is not None
        dumped_params['limit_eval_batches'] = getattr(trainer, f'limit_{stage.dataloader_prefix}_batches')
        dumped_params['loop_verbose'] = loop.verbose
    dumped_params['loop_state_dict'] = deepcopy(loop.state_dict())
    return dumped_params