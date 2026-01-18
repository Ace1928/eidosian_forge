import collections
import logging
from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np
from ray.tune.result import DEFAULT_METRIC
from ray.tune.experiment import Trial
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.util.annotations import PublicAPI
def _trials_beyond_time(self, time: float) -> List[Trial]:
    trials = [trial for trial in self._results if self._results[trial][-1][self._time_attr] >= time]
    return trials