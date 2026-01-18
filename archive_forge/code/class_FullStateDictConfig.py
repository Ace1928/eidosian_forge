from dataclasses import dataclass
from enum import auto, Enum
from typing import Optional, Sequence, Type
import torch
from torch.nn.modules.batchnorm import _BatchNorm
@dataclass
class FullStateDictConfig(StateDictConfig):
    """
    ``FullStateDictConfig`` is a config class meant to be used with
    ``StateDictType.FULL_STATE_DICT``. We recommend enabling both
    ``offload_to_cpu=True`` and ``rank0_only=True`` when saving full state
    dicts to save GPU memory and CPU memory, respectively. This config class
    is meant to be used via the :func:`state_dict_type` context manager as
    follows:

        >>> # xdoctest: +SKIP("undefined variables")
        >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        >>> fsdp = FSDP(model, auto_wrap_policy=...)
        >>> cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        >>> with FSDP.state_dict_type(fsdp, StateDictType.FULL_STATE_DICT, cfg):
        >>>     state = fsdp.state_dict()
        >>>     # `state` will be empty on non rank 0 and contain CPU tensors on rank 0.
        >>> # To reload checkpoint for inference, finetuning, transfer learning, etc:
        >>> model = model_fn() # Initialize model in preparation for wrapping with FSDP
        >>> if dist.get_rank() == 0:
        >>>     # Load checkpoint only on rank 0 to avoid memory redundancy
        >>>     state_dict = torch.load("my_checkpoint.pt")
        >>>     model.load_state_dict(state_dict)
        >>> # All ranks initialize FSDP module as usual. `sync_module_states` argument
        >>> # communicates loaded checkpoint states from rank 0 to rest of the world.
        >>> fsdp = FSDP(model, device_id=torch.cuda.current_device(), auto_wrap_policy=..., sync_module_states=True)
        >>> # After this point, all ranks have FSDP model with loaded checkpoint.

    Attributes:
        rank0_only (bool): If ``True``, then only rank 0 saves the full state
            dict, and nonzero ranks save an empty dict. If ``False``, then all
            ranks save the full state dict. (Default: ``False``)
    """
    rank0_only: bool = False