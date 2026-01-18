import asyncio
import json
import logging
import os
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Set, Tuple
from ray._private.ray_constants import (
import ray._private.runtime_env.agent.runtime_env_consts as runtime_env_consts
from ray._private.ray_logging import setup_component_logger
from ray._private.runtime_env.conda import CondaPlugin
from ray._private.runtime_env.container import ContainerManager
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.java_jars import JavaJarsPlugin
from ray._private.runtime_env.pip import PipPlugin
from ray._private.gcs_utils import GcsAioClient
from ray._private.runtime_env.plugin import (
from ray._private.utils import get_or_create_event_loop
from ray._private.runtime_env.plugin import RuntimeEnvPluginManager
from ray._private.runtime_env.py_modules import PyModulesPlugin
from ray._private.runtime_env.working_dir import WorkingDirPlugin
from ray._private.runtime_env.nsight import NsightPlugin
from ray._private.runtime_env.mpi import MPIPlugin
from ray.core.generated import (
from ray.core.generated.runtime_env_common_pb2 import (
from ray.runtime_env import RuntimeEnv, RuntimeEnvConfig
def _decrease_reference_for_runtime_env(self, serialized_env: str):
    default_logger.debug(f'Decrease reference for runtime env {serialized_env}.')
    unused = False
    if self._runtime_env_reference[serialized_env] > 0:
        self._runtime_env_reference[serialized_env] -= 1
        if self._runtime_env_reference[serialized_env] == 0:
            unused = True
            del self._runtime_env_reference[serialized_env]
    else:
        default_logger.warn(f'Runtime env {serialized_env} does not exist.')
    if unused:
        default_logger.info(f'Unused runtime env {serialized_env}.')
        self._unused_runtime_env_callback(serialized_env)
    return unused