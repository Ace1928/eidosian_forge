import copy
import inspect
import os
import warnings
from contextlib import nullcontext
from typing import Optional, Tuple
import accelerate
import torch
from accelerate.hooks import add_hook_to_module, remove_hook_from_module
from accelerate.utils import is_npu_available, is_xpu_available
from huggingface_hub import file_exists
from huggingface_hub.utils import EntryNotFoundError, HFValidationError
from safetensors.torch import storage_ptr, storage_size
from ..import_utils import is_auto_gptq_available, is_torch_tpu_available
from .constants import (
class ModulesToSaveWrapper(torch.nn.Module):

    def __init__(self, module_to_save, adapter_name):
        super().__init__()
        self.original_module = module_to_save
        self.modules_to_save = torch.nn.ModuleDict({})
        self._active_adapter = adapter_name
        self._disable_adapters = False
        self.update(adapter_name)
        self.check_module()

    def check_module(self):
        """Perform some sanity checks on the module to ensure that it works"""
        forbidden_classes = (torch.nn.ModuleDict, torch.nn.ModuleList, torch.nn.ParameterDict, torch.nn.ParameterList)
        if isinstance(self.original_module, forbidden_classes):
            cls_name = self.original_module.__class__.__name__
            raise TypeError(f'modules_to_save cannot be applied to modules of type {cls_name}')

    @property
    def disable_adapters(self) -> bool:
        return self._disable_adapters

    @property
    def active_adapter(self) -> str:
        return self._active_adapter

    @property
    def weight(self):
        if self.active_adapter not in self.modules_to_save:
            return self.original_module.weight
        return self.modules_to_save[self.active_adapter].weight

    def update(self, adapter_name):
        context_manager = nullcontext()
        for _, param in self.original_module.named_parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param, 'ds_numel'):
                import deepspeed
                context_manager = deepspeed.zero.GatheredParameters(self.original_module.parameters(), modifier_rank=0)
                break
        with context_manager:
            self.modules_to_save.update(torch.nn.ModuleDict({adapter_name: copy.deepcopy(self.original_module)}))
        if hasattr(self.modules_to_save[adapter_name], '_hf_hook'):
            old_hook = self.modules_to_save[adapter_name]._hf_hook
            new_hook = self._create_new_hook(old_hook)
            remove_hook_from_module(self.modules_to_save[adapter_name])
            add_hook_to_module(self.modules_to_save[adapter_name], new_hook)
        self.original_module.requires_grad_(False)
        if adapter_name == self.active_adapter:
            self.modules_to_save[adapter_name].requires_grad_(True)

    def _create_new_hook(self, old_hook):
        """
        Creates a new hook based on the old hook. Use it only if you know what you are doing !
        """
        old_hook_cls = getattr(accelerate.hooks, old_hook.__class__.__name__)
        old_hook_attr = old_hook.__dict__
        filtered_old_hook_attr = {}
        old_hook_init_signature = inspect.signature(old_hook_cls.__init__)
        for k in old_hook_attr.keys():
            if k in old_hook_init_signature.parameters:
                filtered_old_hook_attr[k] = old_hook_attr[k]
        new_hook = old_hook_cls(**filtered_old_hook_attr)
        return new_hook

    def forward(self, *args, **kwargs):
        if self.disable_adapters or self.active_adapter not in self.modules_to_save:
            return self.original_module(*args, **kwargs)
        return self.modules_to_save[self.active_adapter](*args, **kwargs)

    def enable_adapters(self, enabled: bool):
        """Toggle the enabling and disabling of adapters

        Takes care of setting the requires_grad flag for the adapter weights.

        Args:
            enabled (bool): True to enable adapters, False to disable adapters
        """
        if self._disable_adapters is not enabled:
            return
        if enabled:
            self.original_module.requires_grad_(False)
            self.modules_to_save[self.active_adapter].requires_grad_(True)
            self._disable_adapters = False
        else:
            self.original_module.requires_grad_(True)
            self.modules_to_save.requires_grad_(False)
            self._disable_adapters = True

    def set_adapter(self, adapter_name: str):
        """Set the active adapter

        Additionally, this function will set the specified adapter to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (str): The name of the adapter to set as active
        """
        if adapter_name not in self.modules_to_save:
            raise ValueError(f'Adapter {adapter_name} not found in {self.modules_to_save.keys()}')
        self.modules_to_save[self.active_adapter].requires_grad_(False)
        self.modules_to_save[adapter_name].requires_grad_(True)
        self._active_adapter = adapter_name