import warnings
from typing import Dict, List, Optional, Union
import ray
from ray._private.auto_init_hook import auto_init_ray
from ray._private.client_mode_hook import client_mode_should_convert, client_mode_wrap
from ray._private.utils import hex_to_binary, get_ray_doc_version
from ray._raylet import PlacementGroupID
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
@PublicAPI
class PlacementGroup:
    """A handle to a placement group."""

    @staticmethod
    def empty() -> 'PlacementGroup':
        return PlacementGroup(PlacementGroupID.nil())

    def __init__(self, id: 'ray._raylet.PlacementGroupID', bundle_cache: Optional[List[Dict]]=None):
        self.id = id
        self.bundle_cache = bundle_cache

    @property
    def is_empty(self):
        return self.id.is_nil()

    def ready(self) -> 'ray._raylet.ObjectRef':
        """Returns an ObjectRef to check ready status.

        This API runs a small dummy task to wait for placement group creation.
        It is compatible to ray.get and ray.wait.

        Example:
            .. testcode::

                import ray

                pg = ray.util.placement_group([{"CPU": 1}])
                ray.get(pg.ready())

                pg = ray.util.placement_group([{"CPU": 1}])
                ray.wait([pg.ready()])

        """
        self._fill_bundle_cache_if_needed()
        _export_bundle_reservation_check_method_if_needed()
        assert len(self.bundle_cache) != 0, f'ready() cannot be called on placement group object with a bundle length == 0, current bundle length: {len(self.bundle_cache)}'
        return bundle_reservation_check.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=self), resources={BUNDLE_RESOURCE_LABEL: 0.001}).remote(self)

    def wait(self, timeout_seconds: Union[float, int]=30) -> bool:
        """Wait for the placement group to be ready within the specified time.
        Args:
             timeout_seconds(float|int): Timeout in seconds.
        Return:
             True if the placement group is created. False otherwise.
        """
        return _call_placement_group_ready(self.id, timeout_seconds)

    @property
    def bundle_specs(self) -> List[Dict]:
        """List[Dict]: Return bundles belonging to this placement group."""
        self._fill_bundle_cache_if_needed()
        return self.bundle_cache

    @property
    def bundle_count(self) -> int:
        self._fill_bundle_cache_if_needed()
        return len(self.bundle_cache)

    def _fill_bundle_cache_if_needed(self) -> None:
        if not self.bundle_cache:
            self.bundle_cache = _get_bundle_cache(self.id)

    def __eq__(self, other):
        if not isinstance(other, PlacementGroup):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)