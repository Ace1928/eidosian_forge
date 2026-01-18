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
class BatchQueue(Queue):

    def __init__(self, maxsize: int=0, actor_options: Optional[Dict]=None) -> None:
        actor_options = actor_options or {}
        self.maxsize = maxsize
        self.actor = ray.remote(_BatchQueueActor).options(**actor_options).remote(self.maxsize)

    def get_batch(self, batch_size: int=None, total_timeout: Optional[float]=None, first_timeout: Optional[float]=None) -> List[Any]:
        """Gets batch of items from the queue and returns them in a
        list in order.

        Raises:
            Empty: if the queue does not contain the desired number of items
        """
        return ray.get(self.actor.get_batch.remote(batch_size, total_timeout, first_timeout))