import collections
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import logging
from ray.util.annotations import PublicAPI
from ray.tune.result import DEFAULT_METRIC
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.experiment import Trial
from ray.tune.error import TuneError
def _create_bracket(self, s):
    return _Bracket(time_attr=self._time_attr, max_trials=self._get_n0(s), init_t_attr=self._get_r0(s), max_t_attr=self._max_t_attr, eta=self._eta, s=s, stop_last_trials=self._stop_last_trials)