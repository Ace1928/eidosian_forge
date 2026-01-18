from typing import List, MutableSequence, Optional, Tuple, Union
import torch
from lightning_fabric.utilities.exceptions import MisconfigurationException
from lightning_fabric.utilities.types import _DEVICE
def _check_data_type(device_ids: object) -> None:
    """Checks that the device_ids argument is one of the following: int, string, or sequence of integers.

    Args:
        device_ids: gpus/tpu_cores parameter as passed to the Trainer

    Raises:
        TypeError:
            If ``device_ids`` of GPU/TPUs aren't ``int``, ``str`` or sequence of ``int```

    """
    msg = 'Device IDs (GPU/TPU) must be an int, a string, a sequence of ints, but you passed'
    if device_ids is None:
        raise TypeError(f'{msg} None')
    if isinstance(device_ids, (MutableSequence, tuple)):
        for id_ in device_ids:
            id_type = type(id_)
            if id_type is not int:
                raise TypeError(f'{msg} a sequence of {type(id_).__name__}.')
    elif type(device_ids) not in (int, str):
        raise TypeError(f'{msg} {device_ids!r}.')