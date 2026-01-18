from typing import List, MutableSequence, Optional, Tuple, Union
import torch
from lightning_fabric.utilities.exceptions import MisconfigurationException
from lightning_fabric.utilities.types import _DEVICE
Checks that the device_ids argument is one of the following: int, string, or sequence of integers.

    Args:
        device_ids: gpus/tpu_cores parameter as passed to the Trainer

    Raises:
        TypeError:
            If ``device_ids`` of GPU/TPUs aren't ``int``, ``str`` or sequence of ``int```

    