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
class LycorisLayer(BaseTunerLayer):
    """
    A base layer for LyCORIS like adapters
    """
    other_param_names = ('r', 'alpha', 'scaling', 'rank_dropout', 'module_dropout')

    def __init__(self, base_layer: nn.Module) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.alpha = {}
        self.scaling = {}
        self.rank_dropout = {}
        self.module_dropout = {}
        self._disable_adapters = False
        self.merged_adapters = []

    @property
    @abstractmethod
    def _available_adapters(self) -> set[str]:
        ...

    def _init_empty_weights(self, cls, *args, **kwargs) -> None:
        kwargs = kwargs.copy()
        final_device = kwargs.pop('device', 'cpu')
        cls.__init__(self, *args, device='meta', **kwargs)
        self.to_empty(device=final_device)

    @abstractmethod
    def create_adapter_parameters(self, adapter_name: str, r: int, **kwargs):
        ...

    @abstractmethod
    def _get_delta_activations(self, adapter_name: str, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Activations added on top of the base layer output (i.e. after the base layer forward pass)"""

    @abstractmethod
    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        ...

    def merge(self, safe_merge: bool=False, adapter_names: Optional[list[str]]=None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If `True`, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If `None`, all active adapters will be merged.
                Defaults to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return
        for active_adapter in adapter_names:
            if active_adapter in self._available_adapters:
                base_layer = self.get_base_layer()
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(f'NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken')
                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    @abstractmethod
    def reset_adapter_parameters(self, adapter_name: str):
        ...

    def set_scale(self, adapter, scale):
        if adapter not in self._available_adapters:
            return
        self.scaling[adapter] = scale * self.alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return
        for active_adapter in self.active_adapters:
            if active_adapter not in self._available_adapters:
                continue
            self.scaling[active_adapter] *= scale

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn('Already unmerged. Nothing to do.')
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self._available_adapters:
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self._available_adapters:
                continue
            if scale is None:
                self.scaling[active_adapter] = self.alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    @abstractmethod
    def update_layer(self, adapter_name: str, r: int, alpha: float, **kwargs):
        ...