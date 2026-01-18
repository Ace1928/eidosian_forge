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
def _create_rsync_filter_args(self, options):
    rsync_excludes = options.get('rsync_exclude') or []
    rsync_filters = options.get('rsync_filter') or []
    exclude_args = [['--exclude', rsync_exclude] for rsync_exclude in rsync_excludes]
    filter_args = [['--filter', 'dir-merge,- {}'.format(rsync_filter)] for rsync_filter in rsync_filters]
    return [arg for args_list in exclude_args + filter_args for arg in args_list]