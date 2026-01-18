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
def _get_total_available_resources(self, tune_controller: 'TuneController') -> Tuple[float, float]:
    """Get the number of CPUs and GPUs avaialble in total (not just free)"""
    total_available_cpus = tune_controller._resource_updater.get_num_cpus() - self.reserve_resources.get('CPU', 0)
    total_available_gpus = tune_controller._resource_updater.get_num_gpus() - self.reserve_resources.get('GPU', 0)
    return (total_available_cpus, total_available_gpus)