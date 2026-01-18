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
def _check_container_status(self):
    if self.initialized:
        return True
    output = self.ssh_command_runner.run(check_docker_running_cmd(self.container_name, self.docker_cmd), with_output=True).decode('utf-8').strip()
    return 'true' in output.lower() and 'no such object' not in output.lower()