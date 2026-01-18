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
def _wait_for_ip(self, deadline):
    ip = self._get_node_ip()
    if ip is not None:
        cli_logger.labeled_value('Fetched IP', ip)
        return ip
    interval = AUTOSCALER_NODE_SSH_INTERVAL_S
    with cli_logger.group('Waiting for IP'):
        while time.time() < deadline and (not self.provider.is_terminated(self.node_id)):
            ip = self._get_node_ip()
            if ip is not None:
                cli_logger.labeled_value('Received', ip)
                return ip
            cli_logger.print('Not yet available, retrying in {} seconds', cf.bold(str(interval)))
            time.sleep(interval)
    return None