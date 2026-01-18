from typing import Dict, List
import torch.nn as nn
from peft.utils import _freeze_adapter, _get_submodules
from .config import AdaptionPromptConfig, prepare_config
from .layer import AdaptedAttention
from .utils import is_adaption_prompt_trainable
def _set_adapted_attentions(self, adapter_name: str) -> None:
    """Replace LlamaAttention modules with cached AdaptedAttention modules."""
    cached = self._cached_adapters[adapter_name]
    del self._cached_adapters[adapter_name]
    config = self.peft_config[adapter_name]
    for i, par in enumerate(self._parents[adapter_name]):
        setattr(par, config.target_modules, cached[i])