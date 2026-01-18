import copy
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from ...activations import ACT2FN
from ...modeling_outputs import MoECausalLMOutputWithPast, MoEModelOutputWithPastAndCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_gptsan_japanese import GPTSanJapaneseConfig
class GPTSanJapanesePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GPTSanJapaneseConfig
    base_model_prefix = 'gptsan_japanese'
    supports_gradient_checkpointing = False
    _no_split_modules = ['GPTSanJapaneseBlock']
    _skip_keys_device_placement = 'past_key_values'

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {'input_ids': input_ids, 'attention_mask': input_mask}
        return dummy_inputs

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(factor * 1.0)
            module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor * self.config.d_model ** (-0.5))
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, GPTSanJapaneseModel):
            module.embed_tokens.weight.data.normal_(mean=0.0, std=factor * 1.0)
            module.position_embeddings.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, 'extra_position_embeddings') and module.extra_position_embeddings is not None:
                module.extra_position_embeddings.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, (GPTSanJapaneseModel, GPTSanJapaneseForConditionalGeneration)):
            module.final_logits_bias.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, 'lm_head') and (not self.config.tie_word_embeddings):
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, GPTSanJapaneseDenseActDense):
            module.wi.weight.data.normal_(mean=0.0, std=factor * self.config.d_model ** (-0.5))
            if hasattr(module.wi, 'bias') and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * self.config.d_ff ** (-0.5))
            if hasattr(module.wo, 'bias') and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, GPTSanJapaneseAttention):
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_model
            n_heads = self.config.num_heads
            module.k_proj.weight.data.normal_(mean=0.0, std=factor * (d_model * key_value_proj_dim) ** (-0.5))
            module.v_proj.weight.data.normal_(mean=0.0, std=factor * (d_model * key_value_proj_dim) ** (-0.5))
            module.q_proj.weight.data.normal_(mean=0.0, std=factor * (d_model * key_value_proj_dim) ** (-0.5))
            module.out_proj.weight.data.normal_(mean=0.0, std=factor * (n_heads * key_value_proj_dim) ** (-0.5))
        elif isinstance(module, GPTSanJapaneseSparseMLP):
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_model
            n_heads = self.config.num_heads
            module.router.classifier.weight.data.normal_(mean=0.0, std=factor * 1)
            for idx in range(self.config.num_experts):
                module.experts[f'expert_{idx}'].wi.weight.data.normal_(mean=0.0, std=factor * d_model ** (-0.5))
                module.experts[f'expert_{idx}'].wo.weight.data.normal_(mean=0.0, std=factor * d_model ** (-0.5))

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id
        if decoder_start_token_id is None:
            raise ValueError('self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information.')
        if is_torch_fx_proxy(input_ids):
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id
        if pad_token_id is None:
            raise ValueError('self.model.config.pad_token_id has to be defined.')
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids