import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
from ray.autoscaler._private.constants import (
from ray.autoscaler.node_launch_exception import NodeLaunchException
class NodeProviderAvailabilityTracker:
    """A thread safe, TTL cache of node provider availability. We don't use
    cachetools.TTLCache because it always sets the expiration time relative to
    insertion time, but in our case, we want entries to expire relative to when
    the node creation was attempted (and entries aren't necessarily added in
    order). We want the entries to expire because the information grows stale
    over time.
    """

    def __init__(self, timer: Callable[[], float]=time.time, ttl: float=AUTOSCALER_NODE_AVAILABILITY_MAX_STALENESS_S):
        """A cache that tracks the availability of nodes and throw away
        entries which have grown too stale.

        Args:
          timer: A function that returns the current time in seconds.
          ttl: The ttl from the insertion timestamp of an entry.
        """
        self.timer = timer
        self.ttl = ttl
        self.store: Dict[str, Tuple[float, NodeAvailabilityRecord]] = {}
        self.lock = threading.RLock()

    def _update_node_availability_requires_lock(self, node_type: str, timestamp: int, node_launch_exception: Optional[NodeLaunchException]) -> None:
        if node_launch_exception is None:
            record = NodeAvailabilityRecord(node_type=node_type, is_available=True, last_checked_timestamp=timestamp, unavailable_node_information=None)
        else:
            info = UnavailableNodeInformation(category=node_launch_exception.category, description=node_launch_exception.description)
            record = NodeAvailabilityRecord(node_type=node_type, is_available=False, last_checked_timestamp=timestamp, unavailable_node_information=info)
        expiration_time = timestamp + self.ttl
        self.store[node_type] = (expiration_time, record)
        self._remove_old_entries()

    def update_node_availability(self, node_type: str, timestamp: int, node_launch_exception: Optional[NodeLaunchException]) -> None:
        """
        Update the availability and details of a single ndoe type.

        Args:
          node_type: The node type.
          timestamp: The timestamp that this information is accurate as of.
          node_launch_exception: Details about why the node launch failed. If
            empty, the node type will be considered available."""
        with self.lock:
            self._update_node_availability_requires_lock(node_type, timestamp, node_launch_exception)

    def summary(self) -> NodeAvailabilitySummary:
        """
        Returns a summary of node availabilities and their staleness.

        Returns
            A summary of node availabilities and their staleness.
        """
        with self.lock:
            self._remove_old_entries()
            return NodeAvailabilitySummary({node_type: record for node_type, (_, record) in self.store.items()})

    def _remove_old_entries(self):
        """Remove any expired entries from the cache."""
        cur_time = self.timer()
        with self.lock:
            for key, (expiration_time, _) in list(self.store.items()):
                if expiration_time < cur_time:
                    del self.store[key]