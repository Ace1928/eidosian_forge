import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ...generation.logits_process import (
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import CausalLMOutputWithPast, MaskedLMOutput
from ...modeling_utils import PreTrainedModel, get_parameter_device
from ...utils import (
from ..auto import AutoModel
from .configuration_bark import (
from .generation_configuration_bark import (
@add_start_docstrings('Bark semantic (or text) model. It shares the same architecture as the coarse model.\n    It is a GPT-2 like autoregressive model with a language modeling head on top.', BARK_MODEL_START_DOCSTRING.format(config='BarkSemanticConfig'))
class BarkSemanticModel(BarkCausalModel):
    base_model_prefix = 'semantic'
    config_class = BarkSemanticConfig

    def generate(self, input_ids: torch.Tensor, semantic_generation_config: BarkSemanticGenerationConfig=None, history_prompt: Optional[Dict[str, torch.Tensor]]=None, attention_mask: Optional[torch.Tensor]=None, **kwargs) -> torch.LongTensor:
        """
        Generates text semantic tokens from an input prompt and an additional optional `Bark` speaker prompt.

        Args:
            input_ids (`Optional[torch.Tensor]` of shape (batch_size, seq_len), *optional*):
                Input ids, i.e tokenized input sentences. Will be truncated up to
                semantic_generation_config.max_input_semantic_length tokens. Note that the output audios will be as
                long as the longest generation among the batch.
            semantic_generation_config (`BarkSemanticGenerationConfig`):
                Generation config indicating how to generate the semantic tokens.
            history_prompt (`Optional[Dict[str,torch.Tensor]]`, *optional*):
                Optional `Bark` speaker prompt.
            attention_mask (`Optional[torch.Tensor]`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
        Returns:
            torch.LongTensor: Output semantic tokens.
        """
        if semantic_generation_config is None:
            raise ValueError('`semantic_generation_config` has to be provided')
        batch_size = input_ids.shape[0]
        max_input_semantic_length = semantic_generation_config.max_input_semantic_length
        input_ids = input_ids + semantic_generation_config.text_encoding_offset
        if attention_mask is not None:
            input_ids = input_ids.masked_fill((1 - attention_mask).bool(), semantic_generation_config.text_pad_token)
        if history_prompt is not None:
            semantic_history = history_prompt['semantic_prompt'][-max_input_semantic_length:]
            semantic_history = nn.functional.pad(semantic_history, (0, max_input_semantic_length - len(semantic_history)), value=semantic_generation_config.semantic_pad_token, mode='constant')
        else:
            semantic_history = torch.tensor([semantic_generation_config.semantic_pad_token] * max_input_semantic_length, dtype=torch.int).to(self.device)
        semantic_history = torch.repeat_interleave(semantic_history[None], batch_size, dim=0)
        infer_array = torch.tensor([[semantic_generation_config.semantic_infer_token]] * batch_size, dtype=torch.int).to(self.device)
        input_embeds = torch.cat([self.input_embeds_layer(input_ids[:, :max_input_semantic_length]) + self.input_embeds_layer(semantic_history[:, :max_input_semantic_length + 1]), self.input_embeds_layer(infer_array)], dim=1)
        tokens_to_suppress = list(range(semantic_generation_config.semantic_vocab_size, semantic_generation_config.semantic_pad_token))
        tokens_to_suppress.extend(list(range(semantic_generation_config.semantic_pad_token + 1, self.config.output_vocab_size)))
        suppress_tokens_logits_processor = SuppressTokensLogitsProcessor(tokens_to_suppress)
        min_eos_p = kwargs.get('min_eos_p', semantic_generation_config.min_eos_p)
        early_stopping_logits_processor = BarkEosPrioritizerLogitsProcessor(eos_token_id=semantic_generation_config.eos_token_id, min_eos_p=min_eos_p)
        semantic_output = super().generate(torch.ones((batch_size, max_input_semantic_length + 1), dtype=torch.int).to(self.device), input_embeds=input_embeds, logits_processor=[suppress_tokens_logits_processor, early_stopping_logits_processor], generation_config=semantic_generation_config, **kwargs)
        semantic_output = semantic_output[:, max_input_semantic_length + 1:]
        return semantic_output