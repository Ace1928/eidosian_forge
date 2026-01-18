import logging
from contextlib import contextmanager
from typing import Dict, Optional, Set
import ray
from ray.tune.error import TuneError
from ray.util.annotations import Deprecated
from ray.util.placement_group import _valid_resource_shape
from ray.util.scheduling_strategies import (
from ray import tune     ->     from ray import train
from ray.train import Checkpoint
def _tune_task_and_actor_launch_hook(fn, resources: Dict[str, float], strategy: Optional[SchedulingStrategyT]):
    """Launch hook to catch nested tasks that can't fit in the placement group.

    This gives users a nice warning in case they launch a nested task in a Tune trial
    without reserving resources in the trial placement group to fit it.
    """
    key = frozenset({(k, v) for k, v in resources.items() if v > 0})
    if not key or key in _checked_resources:
        return
    if not isinstance(strategy, PlacementGroupSchedulingStrategy) or strategy.placement_group is None:
        return
    cur_pg = ray.util.get_current_placement_group()
    if not cur_pg or strategy.placement_group.id != cur_pg.id:
        return
    _checked_resources.add(key)
    pgf = get_trial_resources()
    if pgf.head_bundle_is_empty:
        available_bundles = cur_pg.bundle_specs[0:]
    else:
        available_bundles = cur_pg.bundle_specs[1:]
    if _valid_resource_shape(resources, available_bundles):
        return
    if fn.class_name:
        submitted = 'actor'
        name = fn.module_name + '.' + fn.class_name + '.' + fn.function_name
    else:
        submitted = 'task'
        name = fn.module_name + '.' + fn.function_name
    main_resources = cur_pg.bundle_specs[0]
    resources = {k: float(v) for k, v in resources.items() if v > 0}
    raise TuneError(f'No trial resources are available for launching the {submitted} `{name}`. To resolve this, specify the Tune option:\n\n>  resources_per_trial=tune.PlacementGroupFactory(\n>    [{main_resources}] + [{resources}] * N\n>  )\n\nWhere `N` is the number of slots to reserve for trial {submitted}s. If you are using a Ray training library, there might be a utility function to set this automatically for you. For more information, refer to https://docs.ray.io/en/latest/tune/tutorials/tune-resources.html')