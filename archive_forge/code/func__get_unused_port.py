import atexit
import json
import logging
import socket
import sys
import time
import traceback
from concurrent import futures
from dataclasses import dataclass
from itertools import chain
import urllib
from threading import Event, Lock, RLock, Thread
from typing import Callable, Dict, List, Optional, Tuple
import grpc
import psutil
import ray
import ray.core.generated.agent_manager_pb2 as agent_manager_pb2
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
import ray.core.generated.runtime_env_agent_pb2 as runtime_env_agent_pb2
from ray._private.client_mode_hook import disable_client_hook
from ray._raylet import GcsClient
from ray._private.parameter import RayParams
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.services import ProcessInfo, start_ray_client_server
from ray._private.tls_utils import add_port_to_grpc_server
from ray._private.utils import detect_fate_sharing_support
from ray.cloudpickle.compat import pickle
from ray.job_config import JobConfig
from ray.util.client.common import (
from ray.util.client.server.dataservicer import _get_reconnecting_from_context
def _get_unused_port(self) -> int:
    """
        Search for a port in _free_ports that is unused.
        """
    with self.server_lock:
        num_ports = len(self._free_ports)
        for _ in range(num_ports):
            port = self._free_ports.pop(0)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.bind(('', port))
            except OSError:
                self._free_ports.append(port)
                continue
            finally:
                s.close()
            return port
    raise RuntimeError('Unable to succeed in selecting a random port.')