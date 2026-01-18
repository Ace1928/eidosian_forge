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
def _call_setup_hook(trainer: 'pl.Trainer') -> None:
    assert trainer.state.fn is not None
    fn = trainer.state.fn
    for module in trainer.lightning_module.modules():
        if isinstance(module, _DeviceDtypeModuleMixin):
            module._device = trainer.strategy.root_device
    for logger in trainer.loggers:
        if hasattr(logger, 'experiment'):
            _ = logger.experiment
    trainer.strategy.barrier('pre_setup')
    if trainer.datamodule is not None:
        _call_lightning_datamodule_hook(trainer, 'setup', stage=fn)
    _call_callback_hooks(trainer, 'setup', stage=fn)
    _call_lightning_module_hook(trainer, 'setup', stage=fn)
    trainer.strategy.barrier('post_setup')