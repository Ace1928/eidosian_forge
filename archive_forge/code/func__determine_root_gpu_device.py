from typing import List, MutableSequence, Optional, Tuple, Union
import torch
from lightning_fabric.utilities.exceptions import MisconfigurationException
from lightning_fabric.utilities.types import _DEVICE
def _determine_root_gpu_device(gpus: List[_DEVICE]) -> Optional[_DEVICE]:
    """
    Args:
        gpus: Non-empty list of ints representing which GPUs to use

    Returns:
        Designated root GPU device id

    Raises:
        TypeError:
            If ``gpus`` is not a list
        AssertionError:
            If GPU list is empty
    """
    if gpus is None:
        return None
    if not isinstance(gpus, list):
        raise TypeError('GPUs should be a list')
    assert len(gpus) > 0, 'GPUs should be a non-empty list'
    return gpus[0]