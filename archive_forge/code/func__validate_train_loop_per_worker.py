import inspect
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union
import ray
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.air.config import RunConfig, ScalingConfig
from ray.train import BackendConfig, Checkpoint, TrainingIterator
from ray.train._internal import session
from ray.train._internal.backend_executor import BackendExecutor, TrialInfo
from ray.train._internal.data_config import DataConfig
from ray.train._internal.session import _TrainingResult, get_session
from ray.train._internal.utils import construct_train_func
from ray.train.trainer import BaseTrainer, GenDataset
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.widgets import Template
from ray.widgets.util import repr_with_fallback
def _validate_train_loop_per_worker(self, train_loop_per_worker: Callable, fn_name: str) -> None:
    num_params = len(inspect.signature(train_loop_per_worker).parameters)
    if num_params > 1:
        raise ValueError(f'{fn_name} should take in 0 or 1 arguments, but it accepts {num_params} arguments instead.')