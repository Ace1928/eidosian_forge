import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_mega import MegaConfig
@add_start_docstrings('MEGA Model with a `language modeling` head on top.', MEGA_START_DOCSTRING)
class MegaForMaskedLM(MegaPreTrainedModel):
    _tied_weights_keys = ['mlm_head.weight']

    def __init__(self, config: MegaConfig):
        super().__init__(config)
        if config.is_decoder:
            logger.warning('If you want to use `MegaForMaskedLM`, set `config.is_decoder=False` for bi-directional self-attention.')
        self.mega = MegaModel(config, add_pooling_layer=False)
        if config.add_lm_hidden_dense_layer:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.hidden_activation = nn.Tanh()
        else:
            self.dense = None
            self.hidden_activation = None
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.post_init()

    def get_output_embeddings(self):
        return self.mlm_head

    def set_output_embeddings(self, new_embeddings):
        self.mlm_head = new_embeddings

    @add_start_docstrings_to_model_forward(MEGA_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC, mask='<mask>', expected_output="' Paris'", expected_loss=0.1)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, encoder_hidden_states: Optional[torch.FloatTensor]=None, encoder_attention_mask: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.mega(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        if self.dense is not None:
            sequence_output = self.dense(sequence_output)
            sequence_output = self.hidden_activation(sequence_output)
        prediction_scores = self.mlm_head(sequence_output)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return MaskedLMOutput(loss=masked_lm_loss, logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)