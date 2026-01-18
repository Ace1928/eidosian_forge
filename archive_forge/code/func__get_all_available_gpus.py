from typing import List, MutableSequence, Optional, Tuple, Union
import torch
from lightning_fabric.utilities.exceptions import MisconfigurationException
from lightning_fabric.utilities.types import _DEVICE
def _get_all_available_gpus(include_cuda: bool=False, include_mps: bool=False) -> List[int]:
    """
    Returns:
        A list of all available GPUs
    """
    from lightning_fabric.accelerators.cuda import _get_all_visible_cuda_devices
    from lightning_fabric.accelerators.mps import _get_all_available_mps_gpus
    cuda_gpus = _get_all_visible_cuda_devices() if include_cuda else []
    mps_gpus = _get_all_available_mps_gpus() if include_mps else []
    return cuda_gpus + mps_gpus