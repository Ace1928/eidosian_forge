import abc
import inspect
import logging
from typing import Tuple, Callable, Optional, Union
from cachetools import LRUCache
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumScript
from pennylane.typing import ResultBatch, TensorLike
def _dev_execute(self, tapes: Batch):
    """
        Converts tapes to numpy before computing just the results on the device.

        Dispatches between the two different device interfaces.
        """
    numpy_tapes = tuple((qml.transforms.convert_to_numpy_parameters(t) for t in tapes))
    if self._uses_new_device:
        return self._device.execute(numpy_tapes, self._execution_config)
    return self._device.batch_execute(numpy_tapes)