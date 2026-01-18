import logging
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Type, Union
from packaging.version import Version
import pytorch_lightning as pl
from lightning_fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from pytorch_lightning.callbacks import Checkpoint, EarlyStopping
from pytorch_lightning.trainer.states import TrainerStatus
from pytorch_lightning.utilities.exceptions import _TunerExitException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def _call_configure_model(trainer: 'pl.Trainer') -> None:
    if is_overridden('configure_sharded_model', trainer.lightning_module):
        with trainer.strategy.model_sharded_context():
            _call_lightning_module_hook(trainer, 'configure_sharded_model')
    if is_overridden('configure_model', trainer.lightning_module):
        with trainer.strategy.tensor_init_context(), trainer.strategy.model_sharded_context(), trainer.precision_plugin.module_init_context():
            _call_lightning_module_hook(trainer, 'configure_model')