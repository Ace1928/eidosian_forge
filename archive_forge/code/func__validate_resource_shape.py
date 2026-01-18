import warnings
from typing import Dict, List, Optional, Union
import ray
from ray._private.auto_init_hook import auto_init_ray
from ray._private.client_mode_hook import client_mode_should_convert, client_mode_wrap
from ray._private.utils import hex_to_binary, get_ray_doc_version
from ray._raylet import PlacementGroupID
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
def _validate_resource_shape(placement_group, resources, placement_resources, task_or_actor_repr):
    bundles = placement_group.bundle_specs
    resources_valid = _valid_resource_shape(resources, bundles)
    placement_resources_valid = _valid_resource_shape(placement_resources, bundles)
    if not resources_valid:
        raise ValueError(f'Cannot schedule {task_or_actor_repr} with the placement group because the resource request {resources} cannot fit into any bundles for the placement group, {bundles}.')
    if not placement_resources_valid:
        raise ValueError(f'Cannot schedule {task_or_actor_repr} with the placement group because the actor requires {placement_resources.get('CPU', 0)} CPU for creation, but it cannot fit into any bundles for the placement group, {bundles}. Consider creating a placement group with CPU resources.')