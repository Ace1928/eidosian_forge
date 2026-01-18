from typing import Any, List, Optional, Union
import torch
from torch.nn import Module
from typing_extensions import Self, override
def _update_properties(root: torch.nn.Module, device: Optional[torch.device]=None, dtype: Optional[Union[str, torch.dtype]]=None) -> None:
    for module in root.modules():
        if not isinstance(module, _DeviceDtypeModuleMixin):
            continue
        if device is not None:
            module._device = device
        if dtype is not None:
            module._dtype = dtype