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
def _validate_datasets(self, datasets) -> None:
    if not (datasets is None or len(datasets) == 0):
        raise ValueError('MosaicTrainer does not support providing dataset shards                 to `trainer_init_per_worker`. Instead of passing in the dataset into                     MosaicTrainer, define a dataloader and use `prepare_dataloader`                     inside the `trainer_init_per_worker`.')