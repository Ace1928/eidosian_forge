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
class BatchMetricsLogger:
    """
    The BatchMetricsLogger will log metrics in batch against an mlflow run.
    If run_id is passed to to constructor then all recording and logging will
    happen against that run_id.
    If no run_id is passed into constructor, then the run ID will be fetched
    from `mlflow.active_run()` each time `record_metrics()` or `flush()` is called; in this
    case, callers must ensure that an active run is present before invoking
    `record_metrics()` or `flush()`.
    """

    def __init__(self, run_id=None, tracking_uri=None):
        self.run_id = run_id
        self.client = MlflowClient(tracking_uri)
        self.data = []
        self.total_training_time = 0
        self.total_log_batch_time = 0
        self.previous_training_timestamp = None

    def flush(self):
        """
        The metrics accumulated by BatchMetricsLogger will be batch logged to an MLflow run.
        """
        self._timed_log_batch()
        self.data = []

    def _timed_log_batch(self):
        current_run_id = mlflow.active_run().info.run_id if self.run_id is None else self.run_id
        start = time.time()
        metrics_slices = [self.data[i:i + MAX_METRICS_PER_BATCH] for i in range(0, len(self.data), MAX_METRICS_PER_BATCH)]
        for metrics_slice in metrics_slices:
            self.client.log_batch(run_id=current_run_id, metrics=metrics_slice)
        end = time.time()
        self.total_log_batch_time += end - start

    def _should_flush(self):
        target_training_to_logging_time_ratio = 10
        if self.total_training_time >= self.total_log_batch_time * target_training_to_logging_time_ratio:
            return True
        return False

    def record_metrics(self, metrics, step=None):
        """
        Submit a set of metrics to be logged. The metrics may not be immediately logged, as this
        class will batch them in order to not increase execution time too much by logging
        frequently.

        Args:
            metrics: Dictionary containing key, value pairs of metrics to be logged.
            step: The training step that the metrics correspond to.
        """
        current_timestamp = time.time()
        if self.previous_training_timestamp is None:
            self.previous_training_timestamp = current_timestamp
        training_time = current_timestamp - self.previous_training_timestamp
        self.total_training_time += training_time
        if step is None:
            step = 0
        for key, value in metrics.items():
            self.data.append(Metric(key, value, int(current_timestamp * 1000), step))
        if self._should_flush():
            self.flush()
        self.previous_training_timestamp = current_timestamp