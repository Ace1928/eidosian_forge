import logging
import math
import time
import warnings
from collections import defaultdict
from typing import Dict, List
import numpy as np
import pandas
import ray
import xgboost as xgb
from ray.util import get_node_ip_address
from modin.core.execution.ray.common import RayWrapper
from modin.distributed.dataframe.pandas import from_partitions
from .utils import RabitContext, RabitContextManager
def _get_min_cpus_per_node():
    """
    Get min number of node CPUs available on cluster nodes.

    Returns
    -------
    int
        Min number of CPUs per node.
    """
    max_node_cpus = min((node.get('Resources', {}).get('CPU', 0.0) for node in ray.nodes()))
    return max_node_cpus if max_node_cpus > 0.0 else _get_cluster_cpus()