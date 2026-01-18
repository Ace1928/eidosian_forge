from collections import defaultdict, deque
from functools import partial
import pathlib
from typing import (
import uuid
import ray
from ray.rllib.core.learner.reduce_result_dict_fn import _reduce_mean_results
from ray.rllib.core.rl_module.rl_module import (
from ray.rllib.core.learner.learner import LearnerSpec
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.actor_manager import FaultTolerantActorManager
from ray.rllib.utils.minibatch_utils import ShardBatchIterator
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.train._internal.backend_executor import BackendExecutor
from ray.tune.utils.file_transfer import sync_dir_between_nodes
from ray.util.annotations import PublicAPI
def _load_module_state(w):
    import ray
    import tempfile
    import shutil
    worker_node_ip = ray.util.get_node_ip_address()
    tmp_marl_module_ckpt_dir = marl_module_ckpt_dir
    tmp_rl_module_ckpt_dirs = rl_module_ckpt_dirs
    if worker_node_ip != head_node_ip:
        if marl_module_ckpt_dir:
            tmp_marl_module_ckpt_dir = tempfile.mkdtemp()
            sync_dir_between_nodes(source_ip=head_node_ip, source_path=marl_module_ckpt_dir, target_ip=worker_node_ip, target_path=tmp_marl_module_ckpt_dir)
        if rl_module_ckpt_dirs:
            tmp_rl_module_ckpt_dirs = {}
            for module_id, path in rl_module_ckpt_dirs.items():
                tmp_rl_module_ckpt_dirs[module_id] = tempfile.mkdtemp()
                sync_dir_between_nodes(source_ip=head_node_ip, source_path=path, target_ip=worker_node_ip, target_path=tmp_rl_module_ckpt_dirs[module_id])
                tmp_rl_module_ckpt_dirs[module_id] = pathlib.Path(tmp_rl_module_ckpt_dirs[module_id])
    if marl_module_ckpt_dir:
        w.module.load_state(tmp_marl_module_ckpt_dir, modules_to_load=modules_to_load)
    if rl_module_ckpt_dirs:
        for module_id, path in tmp_rl_module_ckpt_dirs.items():
            w.module[module_id].load_state(path / RLMODULE_STATE_DIR_NAME)
    if worker_node_ip != head_node_ip:
        if marl_module_ckpt_dir:
            shutil.rmtree(tmp_marl_module_ckpt_dir)
        if rl_module_ckpt_dirs:
            for module_id, path in tmp_rl_module_ckpt_dirs.items():
                shutil.rmtree(path)