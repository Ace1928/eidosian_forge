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
def _set_up_logging_thread(self) -> None:
    """Sets up the logging thread.

        If the logging thread is already set up, this method does nothing.
        """
    with self._lock:
        self._batch_logging_thread = threading.Thread(target=self._logging_loop, name='MLflowAsyncLoggingLoop', daemon=True)
        self._batch_logging_worker_threadpool = ThreadPoolExecutor(max_workers=10, thread_name_prefix='MLflowBatchLoggingWorkerPool')
        self._batch_status_check_threadpool = ThreadPoolExecutor(max_workers=10, thread_name_prefix='MLflowAsyncLoggingStatusCheck')
        self._batch_logging_thread.start()