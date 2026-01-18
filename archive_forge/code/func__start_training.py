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
def _start_training(self, train_func, datasets, metadata, data_config, checkpoint: Optional[Checkpoint]=None):
    tune_session: _TrainSession = get_session()
    assert tune_session, '`_start_training` should only be called from within Tune'
    storage = tune_session.storage
    self._run_with_error_handling(lambda: self._backend_executor.start_training(train_func=train_func, datasets=datasets, metadata=metadata, data_config=data_config, storage=storage, checkpoint=checkpoint))