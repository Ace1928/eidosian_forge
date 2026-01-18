import copy
import datetime
import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Union
import click
import yaml
import ray
from ray._private.usage import usage_lib
from ray.autoscaler._private import subprocess_output_util as cmd_output_util
from ray.autoscaler._private.autoscaler import AutoscalerSummary
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.cluster_dump import (
from ray.autoscaler._private.command_runner import (
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.event_system import CreateClusterEvent, global_event_system
from ray.autoscaler._private.log_timer import LogTimer
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.providers import (
from ray.autoscaler._private.updater import NodeUpdaterThread
from ray.autoscaler._private.util import (
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
from ray.experimental.internal_kv import _internal_kv_put, internal_kv_get_gcs_client
from ray.util.debug import log_once
def _exec(updater: NodeUpdaterThread, cmd: Optional[str]=None, screen: bool=False, tmux: bool=False, port_forward: Optional[Port_forward]=None, with_output: bool=False, run_env: str='auto', shutdown_after_run: bool=False, extra_screen_args: Optional[str]=None) -> str:
    if cmd:
        if screen:
            wrapped_cmd = ['screen', '-L', '-dm']
            if extra_screen_args is not None and len(extra_screen_args) > 0:
                wrapped_cmd += [extra_screen_args]
            wrapped_cmd += ['bash', '-c', quote(cmd + '; exec bash')]
            cmd = ' '.join(wrapped_cmd)
        elif tmux:
            wrapped_cmd = ['tmux', 'new', '-d', 'bash', '-c', quote(cmd + '; exec bash')]
            cmd = ' '.join(wrapped_cmd)
    return updater.cmd_runner.run(cmd, exit_on_fail=True, port_forward=port_forward, with_output=with_output, run_env=run_env, shutdown_after_run=shutdown_after_run)