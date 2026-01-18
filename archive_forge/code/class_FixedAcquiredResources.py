from typing import Dict, List, Optional
from dataclasses import dataclass
import ray
from ray import SCRIPT_MODE, LOCAL_MODE
from ray.air.execution.resources.request import (
from ray.air.execution.resources.resource_manager import ResourceManager
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
@dataclass
class FixedAcquiredResources(AcquiredResources):
    bundles: List[Dict[str, float]]

    def _annotate_remote_entity(self, entity: RemoteRayEntity, bundle: Dict[str, float], bundle_index: int) -> RemoteRayEntity:
        bundle = bundle.copy()
        num_cpus = bundle.pop('CPU', 0)
        num_gpus = bundle.pop('GPU', 0)
        memory = bundle.pop('memory', 0.0)
        return entity.options(num_cpus=num_cpus, num_gpus=num_gpus, memory=memory, resources=bundle)