from __future__ import print_function
import collections
import datetime
import numbers
import os
import sys
import textwrap
import time
import warnings
from typing import Any, Callable, Collection, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import ray
from ray._private.dict import flatten_dict
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.experimental.tqdm_ray import safe_print
from ray.air.util.node import _force_on_current_node
from ray.air.constants import EXPR_ERROR_FILE, TRAINING_ITERATION
from ray.tune.callback import Callback
from ray.tune.logger import pretty_print
from ray.tune.result import (
from ray.tune.experiment.trial import DEBUG_PRINT_INTERVAL, Trial, _Location
from ray.tune.trainable import Trainable
from ray.tune.utils import unflattened_lookup
from ray.tune.utils.log import Verbosity, has_verbosity, set_verbosity
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.queue import Empty, Queue
from ray.widgets import Template
def _get_memory_usage() -> Tuple[float, float, Optional[str]]:
    """Get the current memory consumption.

    Returns:
        Memory used, memory available, and optionally a warning
            message to be shown to the user when memory consumption is higher
            than 90% or if `psutil` is not installed
    """
    try:
        import ray
        import psutil
        total_gb = psutil.virtual_memory().total / 1024 ** 3
        used_gb = total_gb - psutil.virtual_memory().available / 1024 ** 3
        if used_gb > total_gb * 0.9:
            message = ': ***LOW MEMORY*** less than 10% of the memory on this node is available for use. This can cause unexpected crashes. Consider reducing the memory used by your application or reducing the Ray object store size by setting `object_store_memory` when calling `ray.init`.'
        else:
            message = None
        return (round(used_gb, 1), round(total_gb, 1), message)
    except ImportError:
        return (np.nan, np.nan, 'Unknown memory usage. Please run `pip install psutil` to resolve')