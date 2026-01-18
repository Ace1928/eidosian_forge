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
def _auto_configure_shm(self, run_options: List[str]) -> List[str]:
    if self.docker_config.get('disable_shm_size_detection'):
        return run_options
    for run_opt in run_options:
        if '--shm-size' in run_opt:
            logger.info(f'Bypassing automatic SHM-Detection because of `run_option`: {run_opt}')
            return run_options
    try:
        shm_output = self.ssh_command_runner.run('cat /proc/meminfo || true', with_output=True).decode().strip()
        available_memory = int([ln for ln in shm_output.split('\n') if 'MemAvailable' in ln][0].split()[1])
        available_memory_bytes = available_memory * 1024
        shm_size = min(available_memory_bytes * DEFAULT_OBJECT_STORE_MEMORY_PROPORTION * 1.1, DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES)
        return run_options + [f"--shm-size='{shm_size}b'"]
    except Exception as e:
        logger.warning(f'Received error while trying to auto-compute SHM size {e}')
        return run_options