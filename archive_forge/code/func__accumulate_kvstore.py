import time
import logging
import mxnet as mx
from mxnet.module import Module
from .svrg_optimizer import _SVRGOptimizer
def _accumulate_kvstore(self, key, value):
    """Accumulate gradients over all data in the KVStore. In distributed setting, each worker sees a portion of
        data. The full gradients will be aggregated from each worker in the KVStore.

        Parameters
        ----------

        key: int or str
            Key in the KVStore.
        value: NDArray, RowSparseNDArray
            Average of the full gradients.
        """
    self._kvstore.push(key + '_full', value)
    self._kvstore._barrier()
    self._kvstore.pull(key + '_full', value)
    self._allocate_gradients(key, value)