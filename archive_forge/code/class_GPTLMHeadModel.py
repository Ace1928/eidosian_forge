import logging
import math
import re
from collections import OrderedDict, namedtuple
from collections.abc import Sequence
from functools import partial
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import GPT2Config
from flash_attn.models.bigcode import remap_state_dict_hf_bigcode
from flash_attn.models.falcon import remap_state_dict_hf_falcon
from flash_attn.models.gpt_neox import remap_state_dict_hf_gpt_neox
from flash_attn.models.gptj import remap_state_dict_hf_gptj
from flash_attn.models.llama import remap_state_dict_hf_llama
from flash_attn.models.opt import remap_state_dict_hf_opt
from flash_attn.modules.block import Block, ParallelBlock
from flash_attn.modules.embedding import GPT2Embeddings, ParallelGPT2Embeddings
from flash_attn.modules.mha import MHA, ParallelMHA
from flash_attn.modules.mlp import (
from flash_attn.ops.activations import sqrelu_fwd
from flash_attn.utils.distributed import (
from flash_attn.utils.generation import GenerationMixin
from flash_attn.utils.pretrained import state_dict_from_pretrained
class GPTLMHeadModel(GPTPreTrainedModel, GenerationMixin):

    def __init__(self, config: GPT2Config, process_group=None, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(config)
        self.process_group = process_group
        self.transformer = GPTModel(config, process_group=process_group, **factory_kwargs)
        self.tie_word_embeddings = getattr(config, 'tie_word_embeddings', True)
        lm_head_bias = getattr(config, 'lm_head_bias', False)
        pad_vocab_size_multiple = getattr(config, 'pad_vocab_size_multiple', 1)
        vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
        word_embed_proj_dim = getattr(config, 'word_embed_proj_dim', None)
        embed_dim = config.n_embd if word_embed_proj_dim is None else word_embed_proj_dim
        if word_embed_proj_dim is not None:
            self.project_out = nn.Linear(config.n_embd, embed_dim, bias=False, **factory_kwargs)
        else:
            self.project_out = None
        mup_width_scale = getattr(config, 'mup_width_scale', 1.0)
        mup_output_multiplier = getattr(config, 'mup_output_multiplier', 1.0)
        self.output_scale = mup_output_multiplier * mup_width_scale
        if process_group is None:
            self.lm_head = nn.Linear(embed_dim, vocab_size, bias=lm_head_bias, **factory_kwargs)
        else:
            if ColumnParallelLinear is None:
                raise ImportError('fused_dense_lib is not installed')
            self.lm_head = ColumnParallelLinear(embed_dim, vocab_size, process_group, bias=lm_head_bias, sequence_parallel=getattr(config, 'sequence_parallel', True), **factory_kwargs)
        self.norm_head = getattr(config, 'norm_head', False)
        self.apply(partial(_init_weights, n_layer=config.num_hidden_layers, initializer_range=config.initializer_range, mup_width_scale=mup_width_scale))
        self.tie_weights()

    def tie_weights(self):
        if self.tie_word_embeddings:
            self.lm_head.weight = self.transformer.embeddings.word_embeddings.weight
        if self.process_group is not None:
            sync_shared_params(self, self.process_group)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.transformer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0):
        """
        input_ids: (batch, seqlen) int tensor
        inference_params: for generation. Adapted from Megatron-LM (and Apex)
        https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        assert input_ids.ndim == 2, f'Expected `input_ids` to have shape [b, slen], but got shape {input_ids.shape}'
        b, slen = input_ids.shape
        hidden_states = self.transformer(input_ids, position_ids=position_ids, inference_params=inference_params)
        if inference_params is not None:
            assert hidden_states.ndim == 3, 'sequence_parallel is not supported in generation mode'
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
        if self.output_scale != 1.0:
            hidden_states = hidden_states * self.output_scale
        if not self.norm_head:
            lm_logits = self.lm_head(hidden_states)
        else:
            lm_head_weight = F.normalize(self.lm_head.weight)
            if isinstance(self.lm_head, ColumnParallelLinear) and self.lm_head.sequence_parallel:
                hidden_states = all_gather(hidden_states, self.lm_head.process_group)
            lm_logits = F.linear(hidden_states, lm_head_weight, bias=self.lm_head.bias)
        if isinstance(self.lm_head, ColumnParallelLinear) and inference_params is not None:
            lm_logits, _ = all_gather_raw(lm_logits, self.lm_head.process_group)
            lm_logits = rearrange(lm_logits, '(n b) ... d -> b ... (n d)', b=b)
        CausalLMOutput = namedtuple('CausalLMOutput', ['logits'])
        return CausalLMOutput(logits=lm_logits)

    def load_state_dict(self, state_dict, strict=True):
        if 'transformer.ln_0.weight' in state_dict:
            n_layers = len(self.transformer.layers)
            ln_weight = state_dict.pop(f'transformer.layers.{n_layers - 1}.norm2.weight')
            ln_bias = state_dict.pop(f'transformer.layers.{n_layers - 1}.norm2.bias')
            state_dict['transformer.ln_f.weight'] = ln_weight
            state_dict['transformer.ln_f.bias'] = ln_bias
            for l in reversed(range(n_layers)):
                ln_weight = state_dict.pop(f'transformer.layers.{l}.norm1.weight')
                ln_bias = state_dict.pop(f'transformer.layers.{l}.norm1.bias')
                state_dict[f'transformer.layers.{l}.norm2.weight'] = ln_weight
                state_dict[f'transformer.layers.{l}.norm2.bias'] = ln_bias
                if l > 0:
                    ln_weight = state_dict.pop(f'transformer.layers.{l - 1}.norm2.weight')
                    ln_bias = state_dict.pop(f'transformer.layers.{l - 1}.norm2.bias')
                    state_dict[f'transformer.layers.{l}.norm1.weight'] = ln_weight
                    state_dict[f'transformer.layers.{l}.norm1.bias'] = ln_bias
            ln_weight = state_dict.pop('transformer.ln_0.weight')
            ln_bias = state_dict.pop('transformer.ln_0.bias')
            state_dict[f'transformer.layers.0.norm1.weight'] = ln_weight
            state_dict[f'transformer.layers.0.norm1.bias'] = ln_bias
        return super().load_state_dict(state_dict, strict=strict)