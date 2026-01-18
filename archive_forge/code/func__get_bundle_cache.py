import warnings
from typing import Dict, List, Optional, Union
import ray
from ray._private.auto_init_hook import auto_init_ray
from ray._private.client_mode_hook import client_mode_should_convert, client_mode_wrap
from ray._private.utils import hex_to_binary, get_ray_doc_version
from ray._raylet import PlacementGroupID
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
@client_mode_wrap
def _get_bundle_cache(pg_id: PlacementGroupID) -> List[Dict]:
    worker = ray._private.worker.global_worker
    worker.check_connected()
    return list(ray._private.state.state.placement_group_table(pg_id)['bundles'].values())