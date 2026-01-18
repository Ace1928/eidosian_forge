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