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
def _are_bundles_below_limit(self, bundles: List[Dict[str, float]], base_bundles: Optional[List[Dict[str, float]]]=None, max_added_cpus: Optional[float]=None, max_added_gpus: Optional[float]=None):
    if not max_added_cpus:
        if self.increase_by_times > 0:
            max_added_cpus = self.increase_by.get('CPU', 0) * self.increase_by_times
        else:
            max_added_cpus = np.inf
    if not max_added_gpus:
        if self.increase_by_times > 0:
            max_added_gpus = self.increase_by.get('GPU', 0) * self.increase_by_times
        else:
            max_added_gpus = np.inf
    added_resources = self._get_resources_from_bundles(self._get_added_bundles(bundles, base_bundles) if base_bundles else bundles)
    ret = added_resources.get('CPU', -np.inf) < max_added_cpus or added_resources.get('GPU', -np.inf) < max_added_gpus
    return ret