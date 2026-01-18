import importlib
import logging
import os
import pathlib
import random
import sys
import threading
import time
import urllib.parse
from collections import deque
from types import ModuleType
from typing import (
import numpy as np
import ray
from ray._private.utils import _get_pyarrow_version
from ray.data._internal.arrow_ops.transform_pyarrow import unify_schemas
from ray.data.context import WARN_PREFIX, DataContext
def _warn_on_high_parallelism(requested_parallelism, num_read_tasks):
    available_cpu_slots = ray.available_resources().get('CPU', 1)
    if requested_parallelism and num_read_tasks > available_cpu_slots * 4 and (num_read_tasks >= 5000):
        logger.warn(f'{WARN_PREFIX} The requested parallelism of {requested_parallelism} is more than 4x the number of available CPU slots in the cluster of {available_cpu_slots}. This can lead to slowdowns during the data reading phase due to excessive task creation. Reduce the parallelism to match with the available CPU slots in the cluster, or set parallelism to -1 for Ray Data to automatically determine the parallelism. You can ignore this message if the cluster is expected to autoscale.')