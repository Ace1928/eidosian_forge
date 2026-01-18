import functools
from typing import Dict, List, Mapping, Optional, Union
import torch
import torch.nn as nn
from .state import PartialState
from .utils import (
from .utils.modeling import get_non_persistent_buffers
from .utils.other import recursive_getattr
class CpuOffload(ModelHook):
    """
    Offloads a model on the CPU until its forward pass is called. The model will not be offloaded back to the CPU after
    the forward, the user needs to call the `init_hook` method again for this.

    Args:
        execution_device(`str`, `int` or `torch.device`, *optional*):
            The device on which the model should be executed. Will default to the MPS device if it's available, then
            GPU 0 if there is a GPU, and finally to the CPU.
        prev_module_hook (`UserCpuOffloadHook`, *optional*):
            The hook sent back by [`cpu_offload_with_hook`] for a previous model in the pipeline you are running. If
            passed, its offload method will be called just before the forward of the model to which this hook is
            attached.
    """

    def __init__(self, execution_device: Optional[Union[str, int, torch.device]]=None, prev_module_hook: Optional['UserCpuOffloadHook']=None):
        self.prev_module_hook = prev_module_hook
        self.execution_device = execution_device if execution_device is not None else PartialState().default_device

    def init_hook(self, module):
        return module.to('cpu')

    def pre_forward(self, module, *args, **kwargs):
        if self.prev_module_hook is not None:
            self.prev_module_hook.offload()
        module.to(self.execution_device)
        return (send_to_device(args, self.execution_device), send_to_device(kwargs, self.execution_device))