import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from ray.util.annotations import PublicAPI
def _get_proto_job_config(self):
    """Return the protobuf structure of JobConfig."""
    import ray.core.generated.common_pb2 as common_pb2
    from ray._private.utils import get_runtime_env_info
    if self._cached_pb is None:
        pb = common_pb2.JobConfig()
        if self.ray_namespace is None:
            pb.ray_namespace = str(uuid.uuid4())
        else:
            pb.ray_namespace = self.ray_namespace
        pb.jvm_options.extend(self.jvm_options)
        pb.code_search_path.extend(self.code_search_path)
        pb.py_driver_sys_path.extend(self._py_driver_sys_path)
        for k, v in self.metadata.items():
            pb.metadata[k] = v
        parsed_env = self._validate_runtime_env()
        pb.runtime_env_info.CopyFrom(get_runtime_env_info(parsed_env, is_job_runtime_env=True, serialize=False))
        if self._default_actor_lifetime is not None:
            pb.default_actor_lifetime = self._default_actor_lifetime
        self._cached_pb = pb
    return self._cached_pb