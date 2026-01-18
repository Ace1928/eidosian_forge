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
def _estimate_available_parallelism() -> int:
    """Estimates the available CPU parallelism for this Dataset in the cluster.
    If we are currently in a placement group, take that into account."""
    cur_pg = ray.util.get_current_placement_group()
    return _estimate_avail_cpus(cur_pg)