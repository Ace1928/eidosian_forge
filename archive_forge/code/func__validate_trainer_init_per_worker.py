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
def _validate_trainer_init_per_worker(self, trainer_init_per_worker: Callable, fn_name: str) -> None:
    num_params = len(inspect.signature(trainer_init_per_worker).parameters)
    if num_params != 1:
        raise ValueError(f'{fn_name} should take in at most 1 argument (`config`), but it accepts {num_params} arguments instead.')