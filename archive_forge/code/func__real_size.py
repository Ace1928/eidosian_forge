from typing import Any, List
import ray
from ray import cloudpickle
def _real_size(self, item: Any) -> int:
    is_client = ray.util.client.ray.is_connected()
    if is_client:
        return len(cloudpickle.dumps(item))
    global _ray_initialized
    if not _ray_initialized:
        _ray_initialized = True
        ray.put(None)
    return ray._private.worker.global_worker.get_serialization_context().serialize(item).total_bytes