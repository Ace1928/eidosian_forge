import inspect
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type
from composer.loggers.logger_destination import LoggerDestination
from composer.trainer import Trainer
from ray.train import Checkpoint, DataConfig, RunConfig, ScalingConfig
from ray.train.mosaic._mosaic_utils import RayLogger
from ray.train.torch import TorchConfig, TorchTrainer
from ray.train.trainer import GenDataset
from ray.util import PublicAPI
def _validate_trainer_init_config(self, config) -> None:
    if config is not None and 'loggers' in config:
        warnings.warn("Composer's Loggers (any subclass of LoggerDestination) are                 not supported for MosaicComposer. Use Ray provided loggers instead")