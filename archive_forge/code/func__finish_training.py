import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from ray.air._internal.util import StartTraceback
from ray.data import Dataset
from ray.train import Checkpoint, DataConfig
from ray.train._internal.backend_executor import (
from ray.train._internal.session import _TrainingResult, _TrainSession, get_session
from ray.train._internal.utils import ActorWrapper
from ray.train.backend import BackendConfig
from ray.train.base_trainer import (  # noqa: F401
from ray.util.annotations import DeveloperAPI
def _finish_training(self):
    """Finish training and return final results. Propagate any exceptions.

        Blocks until training is finished on all workers.

        Assumes `start_training` has already been called.

        Returns:
            A list of return values from calling ``train_func`` on each worker.
                Each item corresponds to the return value from a single worker.
        """
    return self._backend_executor.finish_training()