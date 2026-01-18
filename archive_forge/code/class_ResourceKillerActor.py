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
class ResourceKillerActor:
    """Abstract base class used to implement resource killers for chaos testing.

    Subclasses should implement _find_resource_to_kill, which should find a resource
    to kill. This method should return the args to _kill_resource, which is another
    abstract method that should kill the resource and add it to the `killed` set.
    """

    def __init__(self, head_node_id, kill_interval_s: float=60, max_to_kill: int=2, kill_filter_fn: Optional[Callable]=None):
        self.kill_interval_s = kill_interval_s
        self.is_running = False
        self.head_node_id = head_node_id
        self.killed = set()
        self.done = ray._private.utils.get_or_create_event_loop().create_future()
        self.max_to_kill = max_to_kill
        self.kill_filter_fn = kill_filter_fn
        self.kill_immediately_after_found = False
        logging.basicConfig(level=logging.INFO)

    def ready(self):
        pass

    async def run(self):
        self.is_running = True
        while self.is_running:
            to_kill = await self._find_resource_to_kill()
            if not self.is_running:
                break
            if self.kill_immediately_after_found:
                sleep_interval = 0
            else:
                sleep_interval = random.random() * self.kill_interval_s
                time.sleep(sleep_interval)
            self._kill_resource(*to_kill)
            if len(self.killed) >= self.max_to_kill:
                break
            await asyncio.sleep(self.kill_interval_s - sleep_interval)
        self.done.set_result(True)

    async def _find_resource_to_kill(self):
        raise NotImplementedError

    def _kill_resource(self, *args):
        raise NotImplementedError

    async def stop_run(self):
        was_running = self.is_running
        self.is_running = False
        return was_running

    async def get_total_killed(self):
        """Get the total number of killed resources"""
        await self.done
        return self.killed