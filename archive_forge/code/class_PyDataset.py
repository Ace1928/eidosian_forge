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
@keras_export(['keras.utils.PyDataset', 'keras.utils.Sequence'])
class PyDataset:
    """Base class for defining a parallel dataset using Python code.

    Every `PyDataset` must implement the `__getitem__()` and the `__len__()`
    methods. If you want to modify your dataset between epochs,
    you may additionally implement `on_epoch_end()`.
    The `__getitem__()` method should return a complete batch
    (not a single sample), and the `__len__` method should return
    the number of batches in the dataset (rather than the number of samples).

    Args:
        workers: Number of workers to use in multithreading or
            multiprocessing.
        use_multiprocessing: Whether to use Python multiprocessing for
            parallelism. Setting this to `True` means that your
            dataset will be replicated in multiple forked processes.
            This is necessary to gain compute-level (rather than I/O level)
            benefits from parallelism. However it can only be set to
            `True` if your dataset can be safely pickled.
        max_queue_size: Maximum number of batches to keep in the queue
            when iterating over the dataset in a multithreaded or
            multipricessed setting.
            Reduce this value to reduce the CPU memory consumption of
            your dataset. Defaults to 10.

    Notes:

    - `PyDataset` is a safer way to do multiprocessing.
        This structure guarantees that the model will only train
        once on each sample per epoch, which is not the case
        with Python generators.
    - The arguments `workers`, `use_multiprocessing`, and `max_queue_size`
        exist to configure how `fit()` uses parallelism to iterate
        over the dataset. They are not being used by the `PyDataset` class
        directly. When you are manually iterating over a `PyDataset`,
        no parallelism is applied.

    Example:

    ```python
    from skimage.io import imread
    from skimage.transform import resize
    import numpy as np
    import math

    # Here, `x_set` is list of path to the images
    # and `y_set` are the associated classes.

    class CIFAR10PyDataset(keras.utils.PyDataset):

        def __init__(self, x_set, y_set, batch_size, **kwargs):
            super().__init__(**kwargs)
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size

        def __len__(self):
            # Return number of batches.
            return math.ceil(len(self.x) / self.batch_size)

        def __getitem__(self, idx):
            # Return x, y for batch idx.
            low = idx * self.batch_size
            # Cap upper bound at array length; the last batch may be smaller
            # if the total number of items is not a multiple of batch size.
            high = min(low + self.batch_size, len(self.x))
            batch_x = self.x[low:high]
            batch_y = self.y[low:high]

            return np.array([
                resize(imread(file_name), (200, 200))
                   for file_name in batch_x]), np.array(batch_y)
    ```
    """

    def __init__(self, workers=1, use_multiprocessing=False, max_queue_size=10):
        self._workers = workers
        self._use_multiprocessing = use_multiprocessing
        self._max_queue_size = max_queue_size

    def _warn_if_super_not_called(self):
        warn = False
        if not hasattr(self, '_workers'):
            self._workers = 1
            warn = True
        if not hasattr(self, '_use_multiprocessing'):
            self._use_multiprocessing = False
            warn = True
        if not hasattr(self, '_max_queue_size'):
            self._max_queue_size = 10
            warn = True
        if warn:
            warnings.warn('Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.', stacklevel=2)

    @property
    def workers(self):
        self._warn_if_super_not_called()
        return self._workers

    @workers.setter
    def workers(self, value):
        self._workers = value

    @property
    def use_multiprocessing(self):
        self._warn_if_super_not_called()
        return self._use_multiprocessing

    @use_multiprocessing.setter
    def use_multiprocessing(self, value):
        self._use_multiprocessing = value

    @property
    def max_queue_size(self):
        self._warn_if_super_not_called()
        return self._max_queue_size

    @max_queue_size.setter
    def max_queue_size(self, value):
        self._max_queue_size = value

    def __getitem__(self, index):
        """Gets batch at position `index`.

        Args:
            index: position of the batch in the PyDataset.

        Returns:
            A batch
        """
        raise NotImplementedError

    def __len__(self):
        """Number of batch in the PyDataset.

        Returns:
            The number of batches in the PyDataset.
        """
        raise NotImplementedError

    def on_epoch_end(self):
        """Method called at the end of every epoch."""
        pass

    def __iter__(self):
        """Create a generator that iterate over the PyDataset."""
        for i in range(len(self)):
            yield self[i]