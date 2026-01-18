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
@ray.remote(num_cpus=0)
class WorkerKillerActor(ResourceKillerActor):

    def __init__(self, head_node_id, kill_interval_s: float=60, max_to_kill: int=2, kill_filter_fn: Optional[Callable]=None):
        super().__init__(head_node_id, kill_interval_s, max_to_kill, kill_filter_fn)
        self.kill_immediately_after_found = True
        from ray.util.state.common import ListApiOptions
        from ray.util.state.api import StateApiClient
        self.client = StateApiClient()
        self.task_options = ListApiOptions(filters=[('state', '=', 'RUNNING'), ('name', '!=', 'WorkerKillActor.run')])

    async def _find_resource_to_kill(self):
        from ray.util.state.common import StateResource
        process_to_kill_task_id = None
        process_to_kill_pid = None
        process_to_kill_node_id = None
        while process_to_kill_pid is None and self.is_running:
            tasks = self.client.list(StateResource.TASKS, options=self.task_options, raise_on_missing_output=False)
            if self.kill_filter_fn is not None:
                tasks = list(filter(self.kill_filter_fn(), tasks))
            for task in tasks:
                if task.worker_id is not None and task.node_id is not None:
                    process_to_kill_task_id = task.task_id
                    process_to_kill_pid = task.worker_pid
                    process_to_kill_node_id = task.node_id
                    break
            await asyncio.sleep(0.1)
        return (process_to_kill_task_id, process_to_kill_pid, process_to_kill_node_id)

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