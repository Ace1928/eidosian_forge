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
def _stream_client_output(remote_future: ray.ObjectRef, progress_reporter: ProgressReporter, string_queue: Queue) -> Any:
    """
    Stream items from string queue to progress_reporter until remote_future resolves
    """
    if string_queue is None:
        return

    def get_next_queue_item():
        try:
            return string_queue.get(block=False)
        except Empty:
            return None

    def _handle_string_queue():
        string_item = get_next_queue_item()
        while string_item is not None:
            progress_reporter.display(string_item)
            string_item = get_next_queue_item()
    while ray.wait([remote_future], timeout=0.2)[1]:
        _handle_string_queue()
    _handle_string_queue()