import logging
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, List, Optional, Sequence, Tuple
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_fabric.strategies.launchers.launcher import _Launcher
from lightning_fabric.utilities.distributed import _set_num_threads_if_needed
from lightning_fabric.utilities.rank_zero import rank_prefixed_message
def _call_children_scripts(self) -> None:
    self._check_can_spawn_children()
    os.environ['MASTER_ADDR'] = self.cluster_environment.main_address
    os.environ['MASTER_PORT'] = str(self.cluster_environment.main_port)
    os.environ['NODE_RANK'] = str(self.cluster_environment.node_rank())
    os.environ['LOCAL_RANK'] = str(self.cluster_environment.local_rank())
    os.environ['WORLD_SIZE'] = f'{self.num_processes * self.num_nodes}'
    for local_rank in range(1, self.num_processes):
        env_copy = os.environ.copy()
        env_copy['LOCAL_RANK'] = f'{local_rank}'
        if os.environ.get('PL_GLOBAL_SEED') is None and 'PL_GLOBAL_SEED' in env_copy:
            del env_copy['PL_GLOBAL_SEED']
        hydra_in_use = False
        cwd: Optional[str] = None
        if _HYDRA_AVAILABLE:
            from hydra.core.hydra_config import HydraConfig
            hydra_in_use = HydraConfig.initialized()
        if hydra_in_use:
            command, cwd = _hydra_subprocess_cmd(local_rank=local_rank)
        else:
            command = _basic_subprocess_cmd()
        proc = subprocess.Popen(command, env=env_copy, cwd=cwd)
        self.procs.append(proc)