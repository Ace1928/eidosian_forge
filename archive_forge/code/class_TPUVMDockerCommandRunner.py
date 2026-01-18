import copy
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, Dict, Optional
from ray._private import ray_constants
from ray.autoscaler._private.command_runner import DockerCommandRunner, SSHCommandRunner
from ray.autoscaler._private.gcp.node import GCPTPUNode
from ray.autoscaler.command_runner import CommandRunnerInterface
from ray.autoscaler.node_provider import NodeProvider
class TPUVMDockerCommandRunner(DockerCommandRunner):
    """A Docker command runner with overwritten IP addresses."""

    def __init__(self, docker_config: Dict[str, Any], internal_ip: str, external_ip: str, worker_id: int, accelerator_type: str, **common_args):
        super().__init__(docker_config=docker_config, **common_args)
        self.ssh_command_runner = TPUVMSSHCommandRunner(internal_ip=internal_ip, external_ip=external_ip, worker_id=worker_id, accelerator_type=accelerator_type, **common_args)