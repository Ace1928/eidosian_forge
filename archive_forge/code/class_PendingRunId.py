import logging
import os
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from typing import Any, Dict, List, Optional, Union
from mlflow.entities import Metric, Param, RunTag
from mlflow.entities.dataset_input import DatasetInput
from mlflow.exceptions import MlflowException
from mlflow.tracking.client import MlflowClient
from mlflow.utils import _truncate_dict, chunk_list
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import (
class PendingRunId:
    """
    Serves as a placeholder for the ID of a run that does not yet exist, enabling additional
    metadata (e.g. metrics, params, ...) to be enqueued for the run prior to its creation.
    """