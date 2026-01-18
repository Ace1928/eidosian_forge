import copy
from enum import Enum
from typing import Any, Dict, List
from ray.autoscaler._private.util import hash_runtime_conf, prepare_config
def _calculate_hashes(self) -> None:
    self._runtime_hash, self._file_mounts_contents_hash = hash_runtime_conf(self._node_configs.get('file_mounts', {}), self._node_configs.get('cluster_synced_files', []), [self._node_configs.get('worker_setup_commands', []), self._node_configs.get('worker_start_ray_commands', [])], generate_file_mounts_contents_hash=self._node_configs.get('generate_file_mounts_contents_hash', True))