import time
import warnings
import threading
import multiprocessing as mp
from .process_executor import ProcessPoolExecutor, EXTRA_QUEUED_CALLS
from .backend.context import cpu_count
from .backend import get_context
def _wait_job_completion(self):
    """Wait for the cache to be empty before resizing the pool."""
    if self._pending_work_items:
        warnings.warn('Trying to resize an executor with running jobs: waiting for jobs completion before resizing.', UserWarning)
        mp.util.debug(f'Executor {self.executor_id} waiting for jobs completion before resizing')
    while self._pending_work_items:
        time.sleep(0.001)