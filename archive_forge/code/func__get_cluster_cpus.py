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
def _get_cluster_cpus():
    """
    Get number of CPUs available on Ray cluster.

    Returns
    -------
    int
        Number of CPUs available on cluster.
    """
    return ray.cluster_resources().get('CPU', 1)