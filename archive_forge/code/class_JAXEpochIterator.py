import collections
import itertools
from functools import partial
import jax
import numpy as np
import tree
from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import ops
from keras.src import optimizers as optimizers_module
from keras.src.backend import distribution_lib as jax_distribution_lib
from keras.src.distribution import distribution_lib
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils
class JAXEpochIterator(EpochIterator):

    def _get_iterator(self):
        return self._prefetch_numpy_iterator(self.data_adapter.get_jax_iterator())

    def _prefetch_numpy_iterator(self, numpy_iterator):
        """Shard and prefetch batches on device.

        Most of the implementation has been borrowed from
        `flax.jax_utils.prefetch_to_device`

        This utility takes an iterator and returns a new iterator which fills an
        on device prefetch buffer. Eager prefetching can improve the performance
        of training loops significantly by overlapping compute and data
        transfer.
        """
        queue = collections.deque()

        def enqueue(n=2):
            for data in itertools.islice(numpy_iterator, n):
                queue.append(_distribute_data(data))
        enqueue(n=2)
        while queue:
            yield queue.popleft()
            enqueue(1)