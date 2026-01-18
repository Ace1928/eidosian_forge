import inspect
import warnings
from functools import wraps, partial
from typing import Callable, Sequence, Optional, Union, Tuple
import logging
from cachetools import LRUCache, Cache
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.typing import ResultBatch
from .set_shots import set_shots
from .jacobian_products import (
def _preprocess_expand_fn(expand_fn: Union[str, Callable], device: device_type, max_expansion: int) -> Callable:
    """Preprocess the ``expand_fn`` configuration property.

    Args:
        expand_fn (str, Callable): If string, then it must be "device".  Otherwise, it should be a map
            from one tape to a new tape. The final tape must be natively executable by the device.
        device (Device, devices.Device): The device that we will be executing on.
        max_expansion (int): The number of times the internal circuit should be expanded when
            executed on a device. Expansion occurs when an operation or measurement is not
            supported, and results in a gate decomposition. If any operations in the decomposition
            remain unsupported by the device, another expansion occurs.

    Returns:
        Callable: a map from one quantum tape to a new one. The output should be compatible with the device.

    """
    if expand_fn != 'device':
        return expand_fn
    if isinstance(device, qml.devices.Device):

        def blank_expansion_function(tape):
            """A blank expansion function since the new device handles expansion in preprocessing."""
            return tape
        return blank_expansion_function

    def device_expansion_function(tape):
        """A wrapper around the device ``expand_fn``."""
        return device.expand_fn(tape, max_expansion=max_expansion)
    return device_expansion_function