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
def _check_docker_installed(self):
    no_exist = 'NoExist'
    output = self.ssh_command_runner.run(f"command -v {self.docker_cmd} || echo '{no_exist}'", with_output=True)
    cleaned_output = output.decode().strip()
    if no_exist in cleaned_output or 'docker' not in cleaned_output:
        if self.docker_cmd == 'docker':
            install_commands = ['curl -fsSL https://get.docker.com -o get-docker.sh', 'sudo sh get-docker.sh', 'sudo usermod -aG docker $USER', 'sudo systemctl restart docker -f']
        else:
            install_commands = ['sudo apt-get update', 'sudo apt-get -y install podman']
        logger.error(f"{self.docker_cmd.capitalize()} not installed. You can install {self.docker_cmd.capitalize()} by adding the following commands to 'initialization_commands':\n" + '\n'.join(install_commands))