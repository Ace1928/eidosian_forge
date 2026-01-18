import csv
import logging
import os
from typing import TYPE_CHECKING, Dict, TextIO
from ray.air.constants import EXPR_PROGRESS_FILE
from ray.tune.logger.logger import _LOGGER_DEPRECATION_WARNING, Logger, LoggerCallback
from ray.tune.utils import flatten_dict
from ray.util.annotations import Deprecated, PublicAPI
def _setup_trial(self, trial: 'Trial'):
    if trial in self._trial_files:
        self._trial_files[trial].close()
    trial.init_local_path()
    local_file = os.path.join(trial.local_path, EXPR_PROGRESS_FILE)
    self._restore_from_remote(EXPR_PROGRESS_FILE, trial)
    self._trial_continue[trial] = os.path.exists(local_file) and os.path.getsize(local_file) > 0
    self._trial_files[trial] = open(local_file, 'at')
    self._trial_csv[trial] = None