import warnings
from typing import Dict, List, Optional, Union
import ray
from ray._private.auto_init_hook import auto_init_ray
from ray._private.client_mode_hook import client_mode_should_convert, client_mode_wrap
from ray._private.utils import hex_to_binary, get_ray_doc_version
from ray._raylet import PlacementGroupID
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
def _valid_resource_shape(resources, bundle_specs):
    """
    If the resource shape cannot fit into every
    bundle spec, return False
    """
    for bundle in bundle_specs:
        fit_in_bundle = True
        for resource, requested_val in resources.items():
            if resource == BUNDLE_RESOURCE_LABEL:
                continue
            if bundle.get(resource, 0) < requested_val:
                fit_in_bundle = False
                break
        if fit_in_bundle:
            return True
    return False