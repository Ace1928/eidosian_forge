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
def _split_data_across_actors(actors: List, set_func, X_parts, y_parts):
    """
    Split row partitions of data between actors.

    Parameters
    ----------
    actors : list
        List of used actors.
    set_func : callable
        The function for setting data in actor.
    X_parts : list
        Row partitions of X data.
    y_parts : list
        Row partitions of y data.
    """
    X_parts_by_actors = _assign_row_partitions_to_actors(actors, X_parts)
    y_parts_by_actors = _assign_row_partitions_to_actors(actors, y_parts, data_for_aligning=X_parts_by_actors)
    for rank, (_, actor) in enumerate(actors):
        set_func(actor, *X_parts_by_actors[rank][0] + y_parts_by_actors[rank][0])