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
def _get_docker_host_mount_location(self, cluster_name: str) -> str:
    """Return the docker host mount directory location."""
    from ray.autoscaler.sdk import get_docker_host_mount_location
    return get_docker_host_mount_location(cluster_name)