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
@DeveloperAPI
class RemoteReporterMixin:
    """Remote reporter abstract mixin class.

    Subclasses of this class will use a Ray Queue to display output
    on the driver side when running Ray Client."""

    @property
    def output_queue(self) -> Queue:
        return getattr(self, '_output_queue', None)

    @output_queue.setter
    def output_queue(self, value: Queue):
        self._output_queue = value

    def display(self, string: str) -> None:
        """Display the progress string.

        Args:
            string: String to display.
        """
        raise NotImplementedError