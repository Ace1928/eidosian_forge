import copy
import io
import os
import math
import logging
from pathlib import Path
from typing import (
import pyarrow.fs
import ray.cloudpickle as pickle
from ray.util import inspect_serializability
from ray.air._internal.uri_utils import URI
from ray.air._internal.usage import AirEntrypoint
from ray.air.config import RunConfig, ScalingConfig
from ray.train._internal.storage import StorageContext, get_fs_and_path
from ray.tune import Experiment, TuneError, ExperimentAnalysis
from ray.tune.execution.experiment_state import _ResumeConfig
from ray.tune.tune import _Config
from ray.tune.registry import is_function_trainable
from ray.tune.result import _get_defaults_results_dir
from ray.tune.result_grid import ResultGrid
from ray.tune.trainable import Trainable
from ray.tune.tune import run
from ray.tune.tune_config import TuneConfig
from ray.tune.utils import flatten_dict
def _validate_param_space_on_restore(self, new_param_space: Dict[str, Any], flattened_param_space_keys: Optional[List[str]]):
    """Determines whether the (optionally) re-specified `param_space` is valid.

        This method performs very loose validation on the new param_space to
        prevent users from trying to specify new hyperparameters to tune over.

        Raises:
            ValueError: if not all keys match the original param_space.
        """
    if flattened_param_space_keys is None:
        return
    keys = sorted(flatten_dict(new_param_space).keys())
    if keys != flattened_param_space_keys:
        raise ValueError(f'Invalid `param_space` input to `Tuner.restore()`. To fix this error, pass in the same `param_space` that was used to initialize the Tuner. Only re-specify the `param_space` to refresh Ray object references that no longer exist due to restoring from a new Ray cluster session. It should not be used to introduce new hyperparameters to tune.\n\nGot: {keys}\nExpected: {flattened_param_space_keys}')