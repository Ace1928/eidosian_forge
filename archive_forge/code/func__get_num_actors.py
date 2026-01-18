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
def _get_num_actors(num_actors=None):
    """
    Get number of actors to create.

    Parameters
    ----------
    num_actors : int, optional
        Desired number of actors. If is None, integer number of actors
        will be computed by condition 2 CPUs per 1 actor.

    Returns
    -------
    int
        Number of actors to create.
    """
    min_cpus_per_node = _get_min_cpus_per_node()
    if num_actors is None:
        num_actors_per_node = max(1, int(min_cpus_per_node // 2))
        return num_actors_per_node * len(ray.nodes())
    elif isinstance(num_actors, int):
        assert num_actors % len(ray.nodes()) == 0, '`num_actors` must be a multiple to number of nodes in Ray cluster.'
        return num_actors
    else:
        RuntimeError('`num_actors` must be int or None')