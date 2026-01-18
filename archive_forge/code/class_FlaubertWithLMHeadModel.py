import itertools
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import gelu
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, SequenceSummary, SQuADHead
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_flaubert import FlaubertConfig
@add_start_docstrings('\n    The Flaubert Model transformer with a language modeling head on top (linear layer with weights tied to the input\n    embeddings).\n    ', FLAUBERT_START_DOCSTRING)
class FlaubertWithLMHeadModel(FlaubertPreTrainedModel):
    _tied_weights_keys = ['pred_layer.proj.weight']

    def __init__(self, config):
        super().__init__(config)
        self.transformer = FlaubertModel(config)
        self.pred_layer = FlaubertPredLayer(config)
        self.post_init()

    def get_output_embeddings(self):
        return self.pred_layer.proj

    def set_output_embeddings(self, new_embeddings):
        self.pred_layer.proj = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        mask_token_id = self.config.mask_token_id
        lang_id = self.config.lang_id
        effective_batch_size = input_ids.shape[0]
        mask_token = torch.full((effective_batch_size, 1), mask_token_id, dtype=torch.long, device=input_ids.device)
        input_ids = torch.cat([input_ids, mask_token], dim=1)
        if lang_id is not None:
            langs = torch.full_like(input_ids, lang_id)
        else:
            langs = None
        return {'input_ids': input_ids, 'langs': langs}

    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC, mask='<special1>')
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, langs: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, lengths: Optional[torch.Tensor]=None, cache: Optional[Dict[str, torch.Tensor]]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, MaskedLMOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, langs=langs, token_type_ids=token_type_ids, position_ids=position_ids, lengths=lengths, cache=cache, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        output = transformer_outputs[0]
        outputs = self.pred_layer(output, labels)
        if not return_dict:
            return outputs + transformer_outputs[1:]
        return MaskedLMOutput(loss=outputs[0] if labels is not None else None, logits=outputs[0] if labels is None else outputs[1], hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)