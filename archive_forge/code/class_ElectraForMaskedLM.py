import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN, get_activation
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_electra import ElectraConfig
@add_start_docstrings('\n    Electra model with a language modeling head on top.\n\n    Even though both the discriminator and generator may be loaded into this model, the generator is the only model of\n    the two to have been trained for the masked language modeling task.\n    ', ELECTRA_START_DOCSTRING)
class ElectraForMaskedLM(ElectraPreTrainedModel):
    _tied_weights_keys = ['generator_lm_head.weight']

    def __init__(self, config):
        super().__init__(config)
        self.electra = ElectraModel(config)
        self.generator_predictions = ElectraGeneratorPredictions(config)
        self.generator_lm_head = nn.Linear(config.embedding_size, config.vocab_size)
        self.post_init()

    def get_output_embeddings(self):
        return self.generator_lm_head

    def set_output_embeddings(self, word_embeddings):
        self.generator_lm_head = word_embeddings

    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint='google/electra-small-generator', output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC, mask='[MASK]', expected_output="'paris'", expected_loss=1.22)
    def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        generator_hidden_states = self.electra(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        generator_sequence_output = generator_hidden_states[0]
        prediction_scores = self.generator_predictions(generator_sequence_output)
        prediction_scores = self.generator_lm_head(prediction_scores)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (prediction_scores,) + generator_hidden_states[1:]
            return (loss,) + output if loss is not None else output
        return MaskedLMOutput(loss=loss, logits=prediction_scores, hidden_states=generator_hidden_states.hidden_states, attentions=generator_hidden_states.attentions)