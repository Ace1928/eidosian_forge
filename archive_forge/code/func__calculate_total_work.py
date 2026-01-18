import collections
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import logging
from ray.util.annotations import PublicAPI
from ray.tune.result import DEFAULT_METRIC
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.experiment import Trial
from ray.tune.error import TuneError
def _calculate_total_work(self, n: int, r: float, s: int):
    work = 0
    cumulative_r = r
    for _ in range(s + 1):
        work += int(n) * int(r)
        n /= self._eta
        n = int(np.ceil(n))
        r *= self._eta
        r = int(min(r, self._max_t_attr - cumulative_r))
    return work