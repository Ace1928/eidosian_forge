from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.llama.configuration_llama import LlamaConfig
class PaddedLlamaForCausalLM(LlamaForCausalLM):
    """Compat layer for padded inputs"""

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids: Optional[torch.Tensor]=None, return_dict: bool=True, output_attentions: bool=False, output_hidden_states: bool=False):
        batch_size, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = int(seqlens_in_batch.max().item())
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
        nz_input_ids = torch.take_along_dim(input_ids, indices)
        nz_position_ids = torch.take_along_dim(position_ids, indices)
        logits = super().forward(nz_input_ids=nz_input_ids, nz_position_ids=nz_position_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen_in_batch).logits
        logits = pad_input(logits, indices, batch_size, seq_len)
        return CausalLMOutputWithPast(logits=logits)

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, **kwargs):
        return {'input_ids': input_ids, 'attention_mask': kwargs.get('attention_mask'), 'position_ids': kwargs.get('position_ids')}