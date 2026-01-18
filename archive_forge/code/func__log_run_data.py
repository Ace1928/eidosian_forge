import atexit
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from mlflow.entities.metric import Metric
from mlflow.entities.param import Param
from mlflow.entities.run_tag import RunTag
from mlflow.utils.async_logging.run_batch import RunBatch
from mlflow.utils.async_logging.run_operations import RunOperations
def _log_run_data(self) -> None:
    """Process the run data in the running runs queues.

        For each run in the running runs queues, this method retrieves the next batch of run data
        from the queue and processes it by calling the `_processing_func` method with the run ID,
        metrics, parameters, and tags in the batch. If the batch is empty, it is skipped. After
        processing the batch, the processed watermark is updated and the batch event is set.
        If an exception occurs during processing, the exception is logged and the batch event is set
        with the exception. If the queue is empty, it is ignored.

        Returns: None
        """
    run_batch = None
    try:
        run_batch = self._queue.get(timeout=1)
    except Empty:
        return

    def logging_func(run_batch):
        try:
            self._logging_func(run_id=run_batch.run_id, metrics=run_batch.metrics, params=run_batch.params, tags=run_batch.tags)
            run_batch.completion_event.set()
        except Exception as e:
            _logger.error(f'Run Id {run_batch.run_id}: Failed to log run data: Exception: {e}')
            run_batch.exception = e
            run_batch.completion_event.set()
    self._batch_logging_worker_threadpool.submit(logging_func, run_batch)