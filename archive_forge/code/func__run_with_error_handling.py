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
def _run_with_error_handling(self, func: Callable):
    try:
        return func()
    except TrainingWorkerError:
        logger.info('Workers have been successfully restarted. Resuming training from latest checkpoint.')
        self._start_training(self._train_func, self._datasets, self._metadata, self._data_config)
        return self._run_with_error_handling(func)
    except InactiveWorkerGroupError:
        raise RuntimeError('This Trainer is not active. It is either shutdown already or never started in the first place. Either create a new Trainer or start this one.') from None
    except TrainBackendError:
        raise RuntimeError('Training failed. You should not be seeing this error and this is a bug. Please create a new issue at https://github.com/ray-project/ray.') from None