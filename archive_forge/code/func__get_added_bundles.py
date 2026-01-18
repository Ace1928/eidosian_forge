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
def _get_added_bundles(self, bundles: List[Dict[str, float]], base_bundles: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Return the difference between bundles and base_bundles"""
    if self.add_bundles:
        added_bundles = bundles[len(base_bundles):]
    else:
        if not bundles:
            bundles = [{'CPU': 0, 'GPU': 0}]
        if not base_bundles:
            base_bundles = [{'CPU': 0, 'GPU': 0}]
        added_bundles = [{'CPU': bundles[0].get('CPU', 0) - base_bundles[0].get('CPU', 0), 'GPU': bundles[0].get('GPU', 0) - base_bundles[0].get('GPU', 0)}]
    return added_bundles