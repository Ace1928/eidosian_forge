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
def _split_list(arr: List[Any], num_splits: int) -> List[List[Any]]:
    """Split the list into `num_splits` lists.

    The splits will be even if the `num_splits` divides the length of list, otherwise
    the remainder (suppose it's R) will be allocated to the first R splits (one for
    each).
    This is the same as numpy.array_split(). The reason we make this a separate
    implementation is to allow the heterogeneity in the elements in the list.
    """
    assert num_splits > 0
    q, r = divmod(len(arr), num_splits)
    splits = [arr[i * q + min(i, r):(i + 1) * q + min(i + 1, r)] for i in range(num_splits)]
    return splits