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
def _create_new_bundles(self, increase_by: Dict[str, float], multiplier: int) -> List[Dict[str, float]]:
    """Create a list of new bundles containing ``increase_by`` * ``multiplier``."""
    multiplier = abs(multiplier)
    if self.add_bundles:
        bundles = [increase_by] * int(multiplier)
    else:
        bundles = [{}]
        bundles[0]['CPU'] = increase_by.get('CPU', 0) * multiplier
        bundles[0]['GPU'] = increase_by.get('GPU', 0) * multiplier
    return bundles