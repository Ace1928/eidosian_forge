from typing import List, MutableSequence, Optional, Tuple, Union
import torch
from lightning_fabric.utilities.exceptions import MisconfigurationException
from lightning_fabric.utilities.types import _DEVICE
def _sanitize_gpu_ids(gpus: List[int], include_cuda: bool=False, include_mps: bool=False) -> List[int]:
    """Checks that each of the GPUs in the list is actually available. Raises a MisconfigurationException if any of the
    GPUs is not available.

    Args:
        gpus: List of ints corresponding to GPU indices

    Returns:
        Unmodified gpus variable

    Raises:
        MisconfigurationException:
            If machine has fewer available GPUs than requested.

    """
    if sum((include_cuda, include_mps)) == 0:
        raise ValueError('At least one gpu type should be specified!')
    all_available_gpus = _get_all_available_gpus(include_cuda=include_cuda, include_mps=include_mps)
    for gpu in gpus:
        if gpu not in all_available_gpus:
            raise MisconfigurationException(f'You requested gpu: {gpus}\n But your machine only has: {all_available_gpus}')
    return gpus