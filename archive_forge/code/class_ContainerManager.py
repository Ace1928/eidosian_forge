import os
import logging
from typing import Optional
from ray._private.runtime_env.context import RuntimeEnvContext
class ContainerManager:

    def __init__(self, tmp_dir: str):
        self._ray_tmp_dir = tmp_dir

    async def setup(self, runtime_env: 'RuntimeEnv', context: RuntimeEnvContext, logger: Optional[logging.Logger]=default_logger):
        if not runtime_env.has_py_container() or not runtime_env.py_container_image():
            return
        container_driver = 'podman'
        context.container = runtime_env['container']
        container_command = [container_driver, 'run', '-v', self._ray_tmp_dir + ':' + self._ray_tmp_dir, '--cgroup-manager=cgroupfs', '--network=host', '--pid=host', '--ipc=host', '--userns=keep-id']
        container_command.append('--env')
        container_command.append('RAY_RAYLET_PID=' + os.getenv('RAY_RAYLET_PID'))
        container_command.append('--env')
        container_command.append('RAY_JOB_ID=$RAY_JOB_ID')
        for env_var_name, env_var_value in os.environ.items():
            if env_var_name.startswith('RAY_') and env_var_name not in ['RAY_RAYLET_PID', 'RAY_JOB_ID']:
                container_command.append('--env')
                container_command.append(f"{env_var_name}='{env_var_value}'")
        if runtime_env.py_container_run_options():
            container_command.extend(runtime_env.py_container_run_options())
        container_command.append('--entrypoint')
        container_command.append('python')
        container_command.append(runtime_env.py_container_image())
        context.py_executable = ' '.join(container_command)
        logger.info('start worker in container with prefix: {}'.format(context.py_executable))