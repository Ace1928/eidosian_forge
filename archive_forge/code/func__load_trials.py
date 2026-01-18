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
def _load_trials(self) -> List[Trial]:
    with self._fs.open_input_stream(self._experiment_json_fs_path) as f:
        experiment_state = json.loads(f.readall(), cls=TuneFunctionDecoder)
    experiment_fs_path = Path(self._experiment_fs_path)
    trials = []
    trial_states = experiment_state['trial_data']
    for trial_json_state, trial_runtime_metadata in trial_states:
        trial = Trial.from_json_state(trial_json_state, stub=True)
        trial.restore_run_metadata(trial_runtime_metadata)
        new_storage = copy.copy(trial.storage)
        new_storage.storage_fs_path = experiment_fs_path.parent.as_posix()
        new_storage.storage_filesystem = self._fs
        new_storage.experiment_dir_name = experiment_fs_path.name
        trial.set_storage(new_storage)
        trials.append(trial)
    return trials