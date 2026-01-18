from __future__ import annotations
import warnings
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Union
import torch
import torch.nn as nn
from tqdm import tqdm
from peft.config import PeftConfig
from peft.utils import (
from .tuners_utils import BaseTuner, BaseTunerLayer, check_adapters_to_merge, check_target_module_exists
class LycorisTuner(BaseTuner):
    """
    A base tuner for LyCORIS like adapters
    """
    prefix: str
    layers_mapping: dict[type[torch.nn.Module], type[LycorisLayer]]

    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    @staticmethod
    def _check_target_module_exists(config, key):
        return check_target_module_exists(config, key)

    @abstractmethod
    def _create_and_replace(self, config: LycorisConfig, adapter_name: str, target: Union[LycorisLayer, nn.Module], target_name, parent, current_key):
        ...

    @classmethod
    def _create_new_module(cls, config: LycorisConfig, adapter_name: str, target: nn.Module, **kwargs) -> LycorisLayer:
        new_module_cls = None
        for subtype, target_cls in cls.layers_mapping.items():
            if hasattr(target, 'base_layer') and isinstance(target.get_base_layer(), subtype) and isinstance(target, BaseTunerLayer):
                new_module_cls = target_cls
                break
            elif isinstance(target, subtype):
                new_module_cls = target_cls
                break
        if new_module_cls is None:
            supported_modules = ', '.join((layer.__name__ for layer in cls.layers_mapping.keys()))
            raise ValueError(f'Target module of type {type(target)} not supported, currently only adapters for {supported_modules} are supported')
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target
        if isinstance(target_base_layer, torch.nn.Conv2d):
            new_module = new_module_cls(target, adapter_name=adapter_name, **kwargs)
        elif isinstance(target_base_layer, torch.nn.Linear):
            new_module = new_module_cls(target, adapter_name=adapter_name, **kwargs)
        else:
            supported_modules = ', '.join((layer.__name__ for layer in cls.layers_mapping.keys()))
            raise ValueError(f'Target module of type {type(target)} not supported, currently only adapters for {supported_modules} are supported')
        return new_module

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            raise ValueError('Please specify `target_modules` in `peft_config`')
        return peft_config

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        if not hasattr(new_module, 'base_layer'):
            new_module.weight = child.weight
            if hasattr(child, 'bias'):
                new_module.bias = child.bias
        if getattr(child, 'state', None) is not None:
            if hasattr(new_module, 'base_layer'):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)
        for name, module in new_module.named_modules():
            if self.prefix in name:
                module.to(child.weight.device)

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def _unload_and_optionally_merge(self, merge: bool=True, progressbar: bool=False, safe_merge: bool=False, adapter_names: Optional[list[str]]=None):
        if merge:
            if getattr(self.model, 'quantization_method', None) == 'gptq':
                raise ValueError('Cannot merge LOHA layers when the model is gptq quantized')
        self._unloading_checks(adapter_names)
        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = 'Unloading ' + ('and merging ' if merge else '') + 'model'
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if hasattr(target, 'base_layer'):
                if merge:
                    target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                self._replace_module(parent, target_name, target.get_base_layer(), target)
            elif isinstance(target, ModulesToSaveWrapper):
                new_module = target.modules_to_save[target.active_adapter]
                if hasattr(new_module, 'base_layer'):
                    if merge:
                        new_module.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    new_module = new_module.get_base_layer()
                setattr(parent, target_name, new_module)
        return self.model

    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        """Disable all adapters.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """
        self._set_adapter_layers(enabled=False)

    def merge_and_unload(self, progressbar: bool=False, safe_merge: bool=False, adapter_names: Optional[list[str]]=None) -> torch.nn.Module:
        """
        This method merges the adapter layers into the base model. This is needed if someone wants to use the base
        model as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.

        """
        return self._unload_and_optionally_merge(progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names)

    def unload(self) -> torch.nn.Module:
        """
        Gets back the base model by removing all the lora modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        for module in self.model.modules():
            if isinstance(module, LycorisLayer):
                if module.merged:
                    warnings.warn('Adapter cannot be set when the model is merged. Unmerging the model first.')
                    module.unmerge()
                module.set_adapter(adapter_name)

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Deletes an existing adapter.

        Args:
            adapter_name (`str`): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f'Adapter {adapter_name} does not exist')
        del self.peft_config[adapter_name]
        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LycorisLayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapters[:]
        self.active_adapter = new_adapter or []