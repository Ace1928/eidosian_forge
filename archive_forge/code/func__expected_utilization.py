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
def _expected_utilization(self, cpus_per_trial, cpus_total):
    num_samples = self._tune_config.num_samples
    if num_samples < 0:
        num_samples = math.inf
    concurrent_trials = self._tune_config.max_concurrent_trials or 0
    if concurrent_trials < 1:
        concurrent_trials = math.inf
    actual_concurrency = min((cpus_total // cpus_per_trial if cpus_per_trial else 0, num_samples, concurrent_trials))
    return actual_concurrency * cpus_per_trial / (cpus_total + 0.001)