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
def _are_resources_the_same(self, trial: Trial, new_resources) -> bool:
    """Returns True if trial's resources are value equal to new_resources.

        Only checks for PlacementGroupFactories at this moment.
        """
    if isinstance(new_resources, PlacementGroupFactory) and trial.placement_group_factory == new_resources:
        logger.debug(f'{trial} PGF {trial.placement_group_factory.required_resources} and {new_resources.required_resources} are the same, skipping')
        return True
    else:
        return False