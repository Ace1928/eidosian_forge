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
def _configure_runtime(self, run_options: List[str]) -> List[str]:
    if self.docker_config.get('disable_automatic_runtime_detection'):
        return run_options
    runtime_output = self.ssh_command_runner.run(f'{self.docker_cmd} ' + "info -f '{{.Runtimes}}' ", with_output=True).decode().strip()
    if 'nvidia-container-runtime' in runtime_output:
        try:
            self.ssh_command_runner.run('nvidia-smi', with_output=False)
            return run_options + ['--runtime=nvidia']
        except Exception as e:
            logger.warning('Nvidia Container Runtime is present, but no GPUs found.')
            logger.debug(f'nvidia-smi error: {e}')
            return run_options
    return run_options