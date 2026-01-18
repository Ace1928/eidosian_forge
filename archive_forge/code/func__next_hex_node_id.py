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
def _next_hex_node_id(self):
    self._next_node_id += 1
    base = 'fffffffffffffffffffffffffffffffffffffffffffffffffff'
    return base + str(self._next_node_id).zfill(5)