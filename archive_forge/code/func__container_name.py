import copy
import json
import logging
import os
import subprocess
import sys
import time
from threading import RLock
from types import ModuleType
from typing import Any, Dict, Optional
import yaml
import ray
import ray._private.ray_constants as ray_constants
from ray.autoscaler._private.fake_multi_node.command_runner import (
from ray.autoscaler.command_runner import CommandRunnerInterface
from ray.autoscaler.node_provider import NodeProvider
from ray.autoscaler.tags import (
def _container_name(self, node_id):
    node_status = self._docker_status.get(node_id, {})
    timeout = time.monotonic() + 60
    while not node_status:
        if time.monotonic() > timeout:
            raise RuntimeError(f'Container for {node_id} never became available.')
        time.sleep(1)
        self._update_docker_status()
        node_status = self._docker_status.get(node_id, {})
    return node_status['Name']