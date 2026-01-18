import abc
import inspect
import logging
from typing import Tuple, Callable, Optional, Union
from cachetools import LRUCache
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumScript
from pennylane.typing import ResultBatch, TensorLike
def _dev_execute_and_compute_derivatives(self, tapes: Batch):
    """
        Converts tapes to numpy before computing the the results and derivatives on the device.

        Dispatches between the two different device interfaces.
        """
    numpy_tapes = tuple((qml.transforms.convert_to_numpy_parameters(t) for t in tapes))
    if self._uses_new_device:
        return self._device.execute_and_compute_derivatives(numpy_tapes, self._execution_config)
    return self._device.execute_and_gradients(numpy_tapes, **self._gradient_kwargs)