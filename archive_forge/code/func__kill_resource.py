import asyncio
from datetime import datetime
import inspect
import fnmatch
import functools
import io
import json
import logging
import math
import os
import pathlib
import random
import socket
import subprocess
import sys
import tempfile
import time
import timeit
import traceback
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, Callable, Dict, List, Optional
import uuid
from dataclasses import dataclass
import requests
from ray._raylet import Config
import psutil  # We must import psutil after ray because we bundle it with ray.
from ray._private import (
from ray._private.worker import RayContext
import yaml
import ray
import ray._private.gcs_utils as gcs_utils
import ray._private.memory_monitor as memory_monitor
import ray._private.services
import ray._private.utils
from ray._private.internal_api import memory_summary
from ray._private.tls_utils import generate_self_signed_tls_certs
from ray._raylet import GcsClientOptions, GlobalStateAccessor
from ray.core.generated import (
from ray.util.queue import Empty, Queue, _QueueActor
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def _kill_resource(self, process_to_kill_task_id, process_to_kill_pid, process_to_kill_node_id):
    if process_to_kill_pid is not None:

        @ray.remote
        def kill_process(pid):
            import psutil
            proc = psutil.Process(pid)
            proc.kill()
        scheduling_strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(node_id=process_to_kill_node_id, soft=False)
        kill_process.options(scheduling_strategy=scheduling_strategy).remote(process_to_kill_pid)
        logging.info(f'Killing pid {process_to_kill_pid} on node {process_to_kill_node_id}')
        self.killed.add((process_to_kill_task_id, process_to_kill_pid))