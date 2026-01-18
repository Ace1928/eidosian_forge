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
def _fit_resume(self, trainable: TrainableType, param_space: Optional[Dict[str, Any]]) -> ExperimentAnalysis:
    """Fitting for a restored Tuner."""
    resume = 'AUTO'
    if self._resume_config:
        if not self._resume_config.resume_unfinished:
            if self._resume_config.resume_errored:
                resume += '+ERRORED_ONLY'
            elif self._resume_config.restart_errored:
                resume += '+RESTART_ERRORED_ONLY'
        elif self._resume_config.resume_errored:
            resume += '+ERRORED'
        elif self._resume_config.restart_errored:
            resume += '+RESTART_ERRORED'
    args = {**self._get_tune_run_arguments(trainable), **dict(run_or_experiment=trainable, config=param_space, resume=resume, search_alg=self._tune_config.search_alg, scheduler=self._tune_config.scheduler), **self._tuner_kwargs}
    analysis = run(**args)
    self.clear_remote_string_queue()
    return analysis