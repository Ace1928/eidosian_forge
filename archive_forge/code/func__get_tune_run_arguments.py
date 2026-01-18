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
def _get_tune_run_arguments(self, trainable: TrainableType) -> Dict[str, Any]:
    """Get tune.run arguments common for both new and resumed runs."""
    checkpoint_config = copy.deepcopy(self._run_config.checkpoint_config)
    if checkpoint_config.checkpoint_frequency:
        handle_checkpoint_freq = getattr(trainable, '_handles_checkpoint_freq', None)
        if handle_checkpoint_freq is False:
            raise ValueError(f'You passed `checkpoint_frequency={checkpoint_config.checkpoint_frequency}` to your CheckpointConfig, but this trainer does not support this argument. If you passed in a Trainer that takes in a custom training loop, you will need to report a checkpoint every `checkpoint_frequency` iterations within your training loop using `ray.train.report(metrics=..., checkpoint=...)` to get this behavior.')
        elif handle_checkpoint_freq is True:
            checkpoint_config.checkpoint_frequency = 0
    if checkpoint_config.checkpoint_at_end is not None:
        handle_cp_at_end = getattr(trainable, '_handles_checkpoint_at_end', None)
        if handle_cp_at_end is False:
            raise ValueError(f'You passed `checkpoint_at_end={checkpoint_config.checkpoint_at_end}` to your CheckpointConfig, but this trainer does not support this argument. If you passed in a Trainer that takes in a custom training loop, you should include one last call to `ray.train.report(metrics=..., checkpoint=...)` at the end of your training loop to get this behavior.')
        elif handle_cp_at_end is True:
            checkpoint_config.checkpoint_at_end = False
    elif is_function_trainable(trainable):
        checkpoint_config.checkpoint_at_end = False
    else:
        checkpoint_config.checkpoint_at_end = True
    return dict(storage_path=self._run_config.storage_path, storage_filesystem=self._run_config.storage_filesystem, name=self._experiment_dir_name, mode=self._tune_config.mode, metric=self._tune_config.metric, callbacks=self._run_config.callbacks, sync_config=self._run_config.sync_config, stop=self._run_config.stop, max_failures=self._run_config.failure_config.max_failures, checkpoint_config=checkpoint_config, raise_on_failed_trial=False, fail_fast=self._run_config.failure_config.fail_fast, progress_reporter=self._run_config.progress_reporter, verbose=self._run_config.verbose, reuse_actors=self._tune_config.reuse_actors, max_concurrent_trials=self._tune_config.max_concurrent_trials, time_budget_s=self._tune_config.time_budget_s, trial_name_creator=self._tune_config.trial_name_creator, trial_dirname_creator=self._tune_config.trial_dirname_creator, _entrypoint=self._entrypoint, local_dir=self._run_config.local_dir, chdir_to_trial_dir=self._tune_config.chdir_to_trial_dir)