import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from getpass import getuser
from shlex import quote
from typing import Dict, List
import click
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.docker import (
from ray.autoscaler._private.log_timer import LogTimer
from ray.autoscaler._private.subprocess_output_util import (
from ray.autoscaler.command_runner import CommandRunnerInterface
def _check_if_container_restart_is_needed(self, image: str, cleaned_bind_mounts: Dict[str, str]) -> bool:
    re_init_required = False
    running_image = self.run(check_docker_image(self.container_name, self.docker_cmd), with_output=True, run_env='host').decode('utf-8').strip()
    if running_image != image:
        cli_logger.error('A container with name {} is running image {} instead ' + 'of {} (which was provided in the YAML)', self.container_name, running_image, image)
    mounts = self.run(check_bind_mounts_cmd(self.container_name, self.docker_cmd), with_output=True, run_env='host').decode('utf-8').strip()
    try:
        active_mounts = json.loads(mounts)
        active_remote_mounts = {mnt['Destination'].strip('/') for mnt in active_mounts}
        requested_remote_mounts = {self._docker_expand_user(remote).strip('/') for remote in cleaned_bind_mounts.keys()}
        unfulfilled_mounts = requested_remote_mounts - active_remote_mounts
        if unfulfilled_mounts:
            re_init_required = True
            cli_logger.warning('This Docker Container is already running. Restarting the Docker container on this node to pick up the following file_mounts {}', unfulfilled_mounts)
    except json.JSONDecodeError:
        cli_logger.verbose('Unable to check if file_mounts specified in the YAML differ from those on the running container.')
    return re_init_required