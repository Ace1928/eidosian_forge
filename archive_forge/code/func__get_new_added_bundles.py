from copy import deepcopy
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Callable, TYPE_CHECKING
import pickle
import warnings
from ray.air.execution.resources.request import _sum_bundles
from ray.util.annotations import PublicAPI
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.experiment import Trial
from ray.tune.execution.placement_groups import PlacementGroupFactory
def _get_new_added_bundles(self, trial: Trial, all_trials: List[Trial], base_bundles: List[Dict[str, float]], increase_by: Dict[str, float], total_available_cpus: float, total_available_gpus: float, used_cpus: float, used_gpus: float) -> List[Dict[str, float]]:
    if self.metric is None:
        raise ValueError('The metric parameter cannot be None. The parameter can be set in either `DistributeResourcesToTopJob`, the base scheduler or in `tune.TuneConfig()` (highest to lowest priority).')
    free_cpus = total_available_cpus - used_cpus
    free_gpus = total_available_gpus - used_gpus
    sorted_trials = sorted(all_trials, key=lambda t: -self._metric_op * t.last_result.get(self.metric, np.inf))
    added_bundles = self._get_added_bundles(trial.placement_group_factory.bundles, base_bundles)
    best_trial = next((t for t in sorted_trials if self._are_bundles_below_limit(t.placement_group_factory.bundles, base_bundles)), sorted_trials[0])
    if trial.trial_id != best_trial.trial_id and self._get_multiplier(increase_by, free_cpus, free_gpus) >= 0:
        return added_bundles
    return self._modify_bundles_with_free_resources(added_bundles, increase_by, free_cpus, free_gpus)