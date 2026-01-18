import asyncio
import logging
import os
import shlex
import subprocess
import sys
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import wandb
from wandb.sdk.launch.environment.abstract import AbstractEnvironment
from wandb.sdk.launch.registry.abstract import AbstractRegistry
from .._project_spec import LaunchProject
from ..builder.build import get_env_vars_dict
from ..errors import LaunchError
from ..utils import (
from .abstract import AbstractRun, AbstractRunner, Status
class LocalContainerRunner(AbstractRunner):
    """Runner class, uses a project to create a LocallySubmittedRun."""

    def __init__(self, api: 'Api', backend_config: Dict[str, Any], environment: AbstractEnvironment, registry: AbstractRegistry) -> None:
        super().__init__(api, backend_config)
        self.environment = environment
        self.registry = registry

    def _populate_docker_args(self, launch_project: LaunchProject, image_uri: str) -> Dict[str, Any]:
        docker_args: Dict[str, Any] = launch_project.fill_macros(image_uri).get('local-container', {})
        if _is_wandb_local_uri(self._api.settings('base_url')):
            if sys.platform == 'win32':
                docker_args['net'] = 'host'
            else:
                docker_args['network'] = 'host'
            if sys.platform == 'linux' or sys.platform == 'linux2':
                docker_args['add-host'] = 'host.docker.internal:host-gateway'
        return docker_args

    async def run(self, launch_project: LaunchProject, image_uri: str) -> Optional[AbstractRun]:
        docker_args = self._populate_docker_args(launch_project, image_uri)
        synchronous: bool = self.backend_config[PROJECT_SYNCHRONOUS]
        env_vars = get_env_vars_dict(launch_project, self._api, MAX_ENV_LENGTHS[self.__class__.__name__])
        if _is_wandb_local_uri(self._api.settings('base_url')) and sys.platform == 'darwin':
            _, _, port = self._api.settings('base_url').split(':')
            env_vars['WANDB_BASE_URL'] = f'http://host.docker.internal:{port}'
        elif _is_wandb_dev_uri(self._api.settings('base_url')):
            env_vars['WANDB_BASE_URL'] = 'http://host.docker.internal:9001'
        if launch_project.docker_image:
            try:
                pull_docker_image(image_uri)
            except Exception as e:
                wandb.termwarn(f'Error attempting to pull docker image {image_uri}')
                if not docker_image_exists(image_uri):
                    raise LaunchError(f'Failed to pull docker image {image_uri} with error: {e}')
            assert launch_project.docker_image == image_uri
        entry_cmd = launch_project.override_entrypoint.command if launch_project.override_entrypoint is not None else None
        command_str = ' '.join(get_docker_command(image_uri, env_vars, docker_args=docker_args, entry_cmd=entry_cmd, additional_args=launch_project.override_args)).strip()
        sanitized_cmd_str = sanitize_wandb_api_key(command_str)
        _msg = f'{LOG_PREFIX}Launching run in docker with command: {sanitized_cmd_str}'
        wandb.termlog(_msg)
        run = _run_entry_point(command_str, launch_project.project_dir)
        if synchronous:
            await run.wait()
        return run