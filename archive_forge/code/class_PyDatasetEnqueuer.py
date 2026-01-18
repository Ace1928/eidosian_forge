import multiprocessing.dummy
import queue
import random
import threading
import time
import warnings
import weakref
from contextlib import closing
import numpy as np
import tree
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
class PyDatasetEnqueuer:
    """Base class to enqueue inputs.

    The task of an Enqueuer is to use parallelism to speed up preprocessing.
    This is done with processes or threads.

    Example:

    ```python
        enqueuer = PyDatasetEnqueuer(...)
        enqueuer.start()
        datas = enqueuer.get()
        for data in datas:
            # Use the inputs; training, evaluating, predicting.
            # ... stop sometime.
        enqueuer.stop()
    ```

    The `enqueuer.get()` should be an infinite stream of data.
    """

    def __init__(self, py_dataset, use_multiprocessing=False):
        self.py_dataset = py_dataset
        self.use_multiprocessing = use_multiprocessing
        global _SEQUENCE_COUNTER
        if _SEQUENCE_COUNTER is None:
            try:
                _SEQUENCE_COUNTER = multiprocessing.Value('i', 0)
            except OSError:
                _SEQUENCE_COUNTER = 0
        if isinstance(_SEQUENCE_COUNTER, int):
            self.uid = _SEQUENCE_COUNTER
            _SEQUENCE_COUNTER += 1
        else:
            with _SEQUENCE_COUNTER.get_lock():
                self.uid = _SEQUENCE_COUNTER.value
                _SEQUENCE_COUNTER.value += 1
        self.workers = 0
        self.executor_fn = None
        self.queue = None
        self.run_thread = None
        self.stop_signal = None

    def is_running(self):
        return self.stop_signal is not None and (not self.stop_signal.is_set())

    def start(self, workers=1, max_queue_size=10):
        """Starts the handler's workers.

        Args:
            workers: Number of workers.
            max_queue_size: queue size
                (when full, workers could block on `put()`)
        """
        if self.use_multiprocessing:
            self.executor_fn = self._get_executor_init(workers)
        else:
            self.executor_fn = lambda _: get_pool_class(False)(workers)
        self.workers = workers
        self.queue = queue.Queue(max_queue_size)
        self.stop_signal = threading.Event()
        self.run_thread = threading.Thread(target=self._run)
        self.run_thread.daemon = True
        self.run_thread.start()

    def _send_py_dataset(self):
        """Sends current Iterable to all workers."""
        _SHARED_SEQUENCES[self.uid] = self.py_dataset

    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called `start()`.

        Args:
            timeout: maximum time to wait on `thread.join()`
        """
        if not self.is_running():
            return
        self.stop_signal.set()
        with self.queue.mutex:
            self.queue.queue.clear()
            self.queue.unfinished_tasks = 0
            self.queue.not_full.notify()
        self.run_thread.join(timeout)
        _SHARED_SEQUENCES[self.uid] = None

    def __del__(self):
        if self.is_running():
            self.stop()

    def _run(self):
        """Submits request to the executor and queue the `Future` objects."""
        raise NotImplementedError

    def _get_executor_init(self, workers):
        """Gets the Pool initializer for multiprocessing.

        Args:
            workers: Number of workers.

        Returns:
            Function, a Function to initialize the pool
        """
        raise NotImplementedError

    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        Returns:
            Generator yielding tuples `(inputs, targets)`
                or `(inputs, targets, sample_weights)`.
        """
        raise NotImplementedError