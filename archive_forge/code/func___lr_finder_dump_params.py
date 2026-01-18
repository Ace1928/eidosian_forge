import importlib
import logging
import os
import uuid
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast
import torch
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.types import _TORCH_LRSCHEDULER
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.parsing import lightning_hasattr, lightning_setattr
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.types import STEP_OUTPUT, LRScheduler, LRSchedulerConfig
def __lr_finder_dump_params(trainer: 'pl.Trainer') -> Dict[str, Any]:
    return {'optimizers': trainer.strategy.optimizers, 'lr_scheduler_configs': trainer.strategy.lr_scheduler_configs, 'callbacks': trainer.callbacks, 'loggers': trainer.loggers, 'max_steps': trainer.fit_loop.max_steps, 'limit_val_batches': trainer.limit_val_batches, 'loop_state_dict': deepcopy(trainer.fit_loop.state_dict())}