from typing import List, MutableSequence, Optional, Tuple, Union
import torch
from lightning_fabric.utilities.exceptions import MisconfigurationException
from lightning_fabric.utilities.types import _DEVICE
def _parse_gpu_ids(gpus: Optional[Union[int, str, List[int]]], include_cuda: bool=False, include_mps: bool=False) -> Optional[List[int]]:
    """Parses the GPU IDs given in the format as accepted by the :class:`~pytorch_lightning.trainer.trainer.Trainer`.

    Args:
        gpus: An int -1 or string '-1' indicate that all available GPUs should be used.
            A list of unique ints or a string containing a list of comma separated unique integers
            indicates specific GPUs to use.
            An int of 0 means that no GPUs should be used.
            Any int N > 0 indicates that GPUs [0..N) should be used.
        include_cuda: A boolean value indicating whether to include CUDA devices for GPU parsing.
        include_mps: A boolean value indicating whether to include MPS devices for GPU parsing.

    Returns:
        A list of GPUs to be used or ``None`` if no GPUs were requested

    Raises:
        MisconfigurationException:
            If no GPUs are available but the value of gpus variable indicates request for GPUs

    .. note::
        ``include_cuda`` and ``include_mps`` default to ``False`` so that you only
        have to specify which device type to use and all other devices are not disabled.

    """
    _check_data_type(gpus)
    if gpus is None or (isinstance(gpus, int) and gpus == 0) or str(gpus).strip() in ('0', '[]'):
        return None
    gpus = _normalize_parse_gpu_string_input(gpus)
    gpus = _normalize_parse_gpu_input_to_list(gpus, include_cuda=include_cuda, include_mps=include_mps)
    if not gpus:
        raise MisconfigurationException('GPUs requested but none are available.')
    if torch.distributed.is_available() and torch.distributed.is_torchelastic_launched() and (len(gpus) != 1) and (len(_get_all_available_gpus(include_cuda=include_cuda, include_mps=include_mps)) == 1):
        return gpus
    _check_unique(gpus)
    return _sanitize_gpu_ids(gpus, include_cuda=include_cuda, include_mps=include_mps)