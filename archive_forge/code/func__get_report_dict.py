from contextlib import contextmanager
from typing import Callable, Dict, List, Union, Optional
import os
import tempfile
import warnings
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.utils import flatten_dict
from ray.util import log_once
from lightgbm.callback import CallbackEnv
from lightgbm.basic import Booster
from ray.util.annotations import Deprecated
def _get_report_dict(self, evals_log: Dict[str, Dict[str, list]]) -> dict:
    result_dict = flatten_dict(evals_log, delimiter='-')
    if not self._metrics:
        report_dict = result_dict
    else:
        report_dict = {}
        for key in self._metrics:
            if isinstance(self._metrics, dict):
                metric = self._metrics[key]
            else:
                metric = key
            report_dict[key] = result_dict[metric]
    if self._results_postprocessing_fn:
        report_dict = self._results_postprocessing_fn(report_dict)
    return report_dict