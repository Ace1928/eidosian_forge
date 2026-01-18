import contextlib
import inspect
import logging
import time
from typing import List
import mlflow
from mlflow.entities import Metric
from mlflow.tracking.client import MlflowClient
from mlflow.utils.validation import MAX_METRICS_PER_BATCH
from mlflow.utils.autologging_utils.client import MlflowAutologgingQueueingClient  # noqa: F401
from mlflow.utils.autologging_utils.events import AutologgingEventLogger
from mlflow.utils.autologging_utils.logging_and_warnings import (
from mlflow.utils.autologging_utils.safety import (  # noqa: F401
from mlflow.utils.autologging_utils.versioning import (
def _timed_log_batch(self):
    current_run_id = mlflow.active_run().info.run_id if self.run_id is None else self.run_id
    start = time.time()
    metrics_slices = [self.data[i:i + MAX_METRICS_PER_BATCH] for i in range(0, len(self.data), MAX_METRICS_PER_BATCH)]
    for metrics_slice in metrics_slices:
        self.client.log_batch(run_id=current_run_id, metrics=metrics_slice)
    end = time.time()
    self.total_log_batch_time += end - start