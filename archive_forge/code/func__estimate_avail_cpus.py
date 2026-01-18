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
def _estimate_avail_cpus(cur_pg: Optional['PlacementGroup']) -> int:
    """Estimates the available CPU parallelism for this Dataset in the cluster.

    If we aren't in a placement group, this is trivially the number of CPUs in the
    cluster. Otherwise, we try to calculate how large the placement group is relative
    to the size of the cluster.

    Args:
        cur_pg: The current placement group, if any.
    """
    cluster_cpus = int(ray.cluster_resources().get('CPU', 1))
    cluster_gpus = int(ray.cluster_resources().get('GPU', 0))
    if cur_pg:
        pg_cpus = 0
        for bundle in cur_pg.bundle_specs:
            cpu_fraction = bundle.get('CPU', 0) / max(1, cluster_cpus)
            gpu_fraction = bundle.get('GPU', 0) / max(1, cluster_gpus)
            max_fraction = max(cpu_fraction, gpu_fraction)
            pg_cpus += 2 * int(max_fraction * cluster_cpus)
        return min(cluster_cpus, pg_cpus)
    return cluster_cpus