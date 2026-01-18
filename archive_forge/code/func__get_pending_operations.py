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
def _get_pending_operations(self, run_id):
    """
        Returns:
            A `_PendingRunOperations` containing all pending operations for the
            specified `run_id`.
        """
    if run_id not in self._pending_ops_by_run_id:
        self._pending_ops_by_run_id[run_id] = _PendingRunOperations(run_id=run_id)
    return self._pending_ops_by_run_id[run_id]