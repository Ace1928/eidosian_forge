import copy
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, Dict, Optional
from ray._private import ray_constants
from ray.autoscaler._private.command_runner import DockerCommandRunner, SSHCommandRunner
from ray.autoscaler._private.gcp.node import GCPTPUNode
from ray.autoscaler.command_runner import CommandRunnerInterface
from ray.autoscaler.node_provider import NodeProvider
class TPUCommandRunner(CommandRunnerInterface):
    """A TPU pod command runner."""

    def __init__(self, instance: GCPTPUNode, log_prefix: str, node_id: str, auth_config: Dict[str, Any], provider: NodeProvider, cluster_name: str, process_runner: ModuleType, use_internal_ip: bool, docker_config: Optional[Dict[str, Any]]=None):

        def create_command_runner(worker_id: int, accelerator_type: str, internal_ip: str, external_ip: str) -> CommandRunnerInterface:
            """Returns the correct base command runner."""
            common_args = {'internal_ip': internal_ip, 'external_ip': external_ip, 'worker_id': worker_id, 'accelerator_type': accelerator_type, 'log_prefix': '[tpu_worker_{}] '.format(worker_id) + log_prefix, 'node_id': node_id, 'provider': provider, 'auth_config': auth_config, 'cluster_name': cluster_name, 'process_runner': process_runner, 'use_internal_ip': use_internal_ip}
            if docker_config and docker_config['container_name'] != '':
                return TPUVMDockerCommandRunner(docker_config=docker_config, **common_args)
            else:
                return TPUVMSSHCommandRunner(**common_args)
        self._command_runners = []
        self._num_workers = instance.num_workers
        for i in range(self._num_workers):
            self._command_runners.append(create_command_runner(worker_id=i, accelerator_type=instance.get('acceleratorType'), internal_ip=instance.get_internal_ip(i), external_ip=instance.get_external_ip(i)))

    @property
    def num_connections(self) -> int:
        """Return the number of active connections allowed at a time.

        We occasionally see issues where too many concurrent connections may lead to
        failed SSH connections when there are too many TPU hosts.

        We utilize this property to cap the maximum number of active connections
        at a time until a proper fix is found.

        """
        num_max_concurrent_active_connections = ray_constants.env_integer(ray_constants.RAY_TPU_MAX_CONCURRENT_CONNECTIONS_ENV_VAR, default=16)
        return min(self._num_workers, num_max_concurrent_active_connections)

    def run(self, cmd, timeout=120, exit_on_fail=False, port_forward=None, with_output=False, environment_variables: Dict[str, object]=None, run_env='auto', ssh_options_override_ssh_key='', shutdown_after_run=False) -> str:
        with ThreadPoolExecutor(self.num_connections) as executor:
            results = executor.map(lambda i: self._command_runners[i].run(cmd=cmd, timeout=timeout, exit_on_fail=exit_on_fail, port_forward=port_forward, with_output=with_output, environment_variables=copy.deepcopy(environment_variables), run_env=run_env, ssh_options_override_ssh_key=ssh_options_override_ssh_key, shutdown_after_run=shutdown_after_run), range(self._num_workers))
        return list(results)[0]

    def run_rsync_up(self, *args, **kwargs) -> None:
        with ThreadPoolExecutor(self.num_connections) as executor:
            executor.map(lambda i: self._command_runners[i].run_rsync_up(*args, **kwargs), range(self._num_workers))

    def run_rsync_down(self, *args, **kwargs) -> None:
        """Rsync files down from the cluster node.

        Args:
            source: The (remote) source directory or file.
            target: The (local) destination path.
        """
        with ThreadPoolExecutor(self.num_connections) as executor:
            executor.map(lambda i: self._command_runners[i].run_rsync_down(*args, **kwargs), range(self._num_workers))

    def remote_shell_command_str(self) -> str:
        """Return the command the user can use to open a shell."""
        return self._command_runners[0].remote_shell_command_str()

    def run_init(self, *args, **kwargs) -> Optional[bool]:
        """Used to run extra initialization commands.

        Args:
            as_head: Run as head image or worker.
            file_mounts: Files to copy to the head and worker nodes.
            sync_run_yet: Whether sync has been run yet.

        Returns:
            optional: Whether initialization is necessary.
        """
        with ThreadPoolExecutor(self.num_connections) as executor:
            results = executor.map(lambda i: self._command_runners[i].run_init(*args, **kwargs), range(self._num_workers))
        return any(results)