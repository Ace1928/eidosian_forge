import abc
import json
from copy import deepcopy
from inspect import signature
from typing import Dict, List, Union
from dataclasses import dataclass
import ray
from ray.util import placement_group
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
@dataclass
class AcquiredResources(abc.ABC):
    """Base class for resources that have been acquired.

    Acquired resources can be associated to Ray objects, which can then be
    scheduled using these resources.

    Internally this can point e.g. to a placement group, a placement
    group bundle index, or just raw resources.

    The main API is the `annotate_remote_entities` method. This will associate
    remote Ray objects (tasks and actors) with the acquired resources by setting
    the Ray remote options to use the acquired resources.
    """
    resource_request: ResourceRequest

    def annotate_remote_entities(self, entities: List[RemoteRayEntity]) -> List[Union[RemoteRayEntity]]:
        """Return remote ray entities (tasks/actors) to use the acquired resources.

        The first entity will be associated with the first bundle, the second
        entity will be associated with the second bundle, etc.

        Args:
            entities: Remote Ray entities to annotate with the acquired resources.
        """
        bundles = self.resource_request.bundles
        num_bundles = len(bundles) + int(self.resource_request.head_bundle_is_empty)
        if len(entities) > num_bundles:
            raise RuntimeError(f'The number of callables to annotate ({len(entities)}) cannot exceed the number of available bundles ({num_bundles}).')
        annotated = []
        if self.resource_request.head_bundle_is_empty:
            annotated.append(self._annotate_remote_entity(entities[0], {}, bundle_index=0))
            entities = entities[1:]
        for i, (entity, bundle) in enumerate(zip(entities, bundles)):
            annotated.append(self._annotate_remote_entity(entity, bundle, bundle_index=i))
        return annotated

    def _annotate_remote_entity(self, entity: RemoteRayEntity, bundle: Dict[str, float], bundle_index: int) -> RemoteRayEntity:
        raise NotImplementedError