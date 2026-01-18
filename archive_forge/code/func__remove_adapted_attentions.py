from typing import Dict, List
import torch.nn as nn
from peft.utils import _freeze_adapter, _get_submodules
from .config import AdaptionPromptConfig, prepare_config
from .layer import AdaptedAttention
from .utils import is_adaption_prompt_trainable
def _remove_adapted_attentions(self, adapter_name: str) -> None:
    """Remove AdaptedAttention modules from the model and store them in the cache."""
    config = self.peft_config[adapter_name]
    adapted_attentions = []
    for par in self._parents[adapter_name]:
        attn = getattr(par, config.target_modules)
        adapted_attentions.append(attn)
        setattr(par, config.target_modules, attn.model)
    self._cached_adapters[adapter_name] = adapted_attentions