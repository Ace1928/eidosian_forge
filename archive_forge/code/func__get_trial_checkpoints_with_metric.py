import copy
import fnmatch
import io
import json
import logging
from numbers import Number
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pyarrow.fs
from ray.util.annotations import PublicAPI
from ray.air.constants import (
from ray.train import Checkpoint
from ray.train._internal.storage import (
from ray.tune.execution.tune_controller import TuneController
from ray.tune.experiment import Trial
from ray.tune.result import (
from ray.tune.utils import flatten_dict
from ray.tune.utils.serialization import TuneFunctionDecoder
from ray.tune.utils.util import is_nan_or_inf, is_nan, unflattened_lookup
def _get_trial_checkpoints_with_metric(self, trial: Trial, metric: Optional[str]=None) -> List[Tuple[Checkpoint, Number]]:
    """Get all checkpoints and a specified metric of a trial.

        Args:
            trial: The log directory of a trial, or a trial instance.
            metric: key for trial info to return, e.g. "mean_accuracy".
                "training_iteration" is used by default if no value was
                passed to ``self.default_metric``.

        Returns:
            List of [Checkpoint, metric] for all checkpoints of the trial.
        """
    metric = metric or self.default_metric or TRAINING_ITERATION
    best_checkpoint_results = trial.run_metadata.checkpoint_manager.best_checkpoint_results
    best_checkpoints = [(checkpoint_result.checkpoint, checkpoint_result.metrics) for checkpoint_result in best_checkpoint_results]
    return [(checkpoint, unflattened_lookup(metric, metrics)) for checkpoint, metrics in best_checkpoints]