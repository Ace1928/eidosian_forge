from typing import Dict, List
import torch.nn as nn
from peft.utils import _freeze_adapter, _get_submodules
from .config import AdaptionPromptConfig, prepare_config
from .layer import AdaptedAttention
from .utils import is_adaption_prompt_trainable
def _create_adapted_attentions(self, config: AdaptionPromptConfig, parents: List[nn.Module]) -> None:
    """Wrap LlamaAttention modules with newly created AdaptedAttention modules."""
    for par in parents:
        attn = AdaptedAttention(model_type=self.model.config.model_type, adapter_len=config.adapter_len, model=getattr(par, config.target_modules))
        setattr(par, config.target_modules, attn)