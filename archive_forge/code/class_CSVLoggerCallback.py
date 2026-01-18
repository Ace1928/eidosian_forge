import csv
import logging
import os
from typing import TYPE_CHECKING, Dict, TextIO
from ray.air.constants import EXPR_PROGRESS_FILE
from ray.tune.logger.logger import _LOGGER_DEPRECATION_WARNING, Logger, LoggerCallback
from ray.tune.utils import flatten_dict
from ray.util.annotations import Deprecated, PublicAPI
@PublicAPI
class CSVLoggerCallback(LoggerCallback):
    """Logs results to progress.csv under the trial directory.

    Automatically flattens nested dicts in the result dict before writing
    to csv:

        {"a": {"b": 1, "c": 2}} -> {"a/b": 1, "a/c": 2}

    """
    _SAVED_FILE_TEMPLATES = [EXPR_PROGRESS_FILE]

    def __init__(self):
        self._trial_continue: Dict['Trial', bool] = {}
        self._trial_files: Dict['Trial', TextIO] = {}
        self._trial_csv: Dict['Trial', csv.DictWriter] = {}

    def _setup_trial(self, trial: 'Trial'):
        if trial in self._trial_files:
            self._trial_files[trial].close()
        trial.init_local_path()
        local_file = os.path.join(trial.local_path, EXPR_PROGRESS_FILE)
        self._restore_from_remote(EXPR_PROGRESS_FILE, trial)
        self._trial_continue[trial] = os.path.exists(local_file) and os.path.getsize(local_file) > 0
        self._trial_files[trial] = open(local_file, 'at')
        self._trial_csv[trial] = None

    def log_trial_result(self, iteration: int, trial: 'Trial', result: Dict):
        if trial not in self._trial_files:
            self._setup_trial(trial)
        tmp = result.copy()
        tmp.pop('config', None)
        result = flatten_dict(tmp, delimiter='/')
        if not self._trial_csv[trial]:
            self._trial_csv[trial] = csv.DictWriter(self._trial_files[trial], result.keys())
            if not self._trial_continue[trial]:
                self._trial_csv[trial].writeheader()
        self._trial_csv[trial].writerow({k: v for k, v in result.items() if k in self._trial_csv[trial].fieldnames})
        self._trial_files[trial].flush()

    def log_trial_end(self, trial: 'Trial', failed: bool=False):
        if trial not in self._trial_files:
            return
        del self._trial_csv[trial]
        self._trial_files[trial].close()
        del self._trial_files[trial]