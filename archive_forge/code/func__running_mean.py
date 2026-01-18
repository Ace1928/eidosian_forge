import collections
import logging
from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np
from ray.tune.result import DEFAULT_METRIC
from ray.tune.experiment import Trial
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.util.annotations import PublicAPI
def _running_mean(self, trial: Trial, time: float) -> np.ndarray:
    results = self._results[trial]
    scoped_results = [r for r in results if self._grace_period <= r[self._time_attr] <= time]
    return np.mean([r[self._metric] for r in scoped_results])