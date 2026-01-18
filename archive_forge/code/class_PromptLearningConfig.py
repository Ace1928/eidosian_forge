import inspect
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Union
from huggingface_hub import hf_hub_download
from transformers.utils import PushToHubMixin
from .utils import CONFIG_NAME, PeftType, TaskType
@dataclass
class PromptLearningConfig(PeftConfig):
    """
    This is the base configuration class to store the configuration of [`PrefixTuning`], [`PromptEncoder`], or
    [`PromptTuning`].

    Args:
        num_virtual_tokens (`int`): The number of virtual tokens to use.
        token_dim (`int`): The hidden embedding dimension of the base transformer model.
        num_transformer_submodules (`int`): The number of transformer submodules in the base transformer model.
        num_attention_heads (`int`): The number of attention heads in the base transformer model.
        num_layers (`int`): The number of layers in the base transformer model.
    """
    num_virtual_tokens: int = field(default=None, metadata={'help': 'Number of virtual tokens'})
    token_dim: int = field(default=None, metadata={'help': 'The hidden embedding dimension of the base transformer model'})
    num_transformer_submodules: Optional[int] = field(default=None, metadata={'help': 'Number of transformer submodules'})
    num_attention_heads: Optional[int] = field(default=None, metadata={'help': 'Number of attention heads'})
    num_layers: Optional[int] = field(default=None, metadata={'help': 'Number of transformer layers'})

    @property
    def is_prompt_learning(self) -> bool:
        """
        Utility method to check if the configuration is for prompt learning.
        """
        return True