from typing import List, MutableSequence, Optional, Tuple, Union
import torch
from lightning_fabric.utilities.exceptions import MisconfigurationException
from lightning_fabric.utilities.types import _DEVICE
def _normalize_parse_gpu_input_to_list(gpus: Union[int, List[int], Tuple[int, ...]], include_cuda: bool, include_mps: bool) -> Optional[List[int]]:
    assert gpus is not None
    if isinstance(gpus, (MutableSequence, tuple)):
        return list(gpus)
    if not gpus:
        return None
    if gpus == -1:
        return _get_all_available_gpus(include_cuda=include_cuda, include_mps=include_mps)
    return list(range(gpus))