import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN, gelu
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_luke import LukeConfig
@add_start_docstrings('\n    The LUKE model with a language modeling head and entity prediction head on top for masked language modeling and\n    masked entity prediction.\n    ', LUKE_START_DOCSTRING)
class LukeForMaskedLM(LukePreTrainedModel):
    _tied_weights_keys = ['lm_head.decoder.weight', 'lm_head.decoder.bias', 'entity_predictions.decoder.weight']

    def __init__(self, config):
        super().__init__(config)
        self.luke = LukeModel(config)
        self.lm_head = LukeLMHead(config)
        self.entity_predictions = EntityPredictionHead(config)
        self.loss_fn = nn.CrossEntropyLoss()
        self.post_init()

    def tie_weights(self):
        super().tie_weights()
        self._tie_or_clone_weights(self.entity_predictions.decoder, self.luke.entity_embeddings.entity_embeddings)

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(LUKE_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=LukeMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, entity_ids: Optional[torch.LongTensor]=None, entity_attention_mask: Optional[torch.LongTensor]=None, entity_token_type_ids: Optional[torch.LongTensor]=None, entity_position_ids: Optional[torch.LongTensor]=None, labels: Optional[torch.LongTensor]=None, entity_labels: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, LukeMaskedLMOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        entity_labels (`torch.LongTensor` of shape `(batch_size, entity_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.luke(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, entity_ids=entity_ids, entity_attention_mask=entity_attention_mask, entity_token_type_ids=entity_token_type_ids, entity_position_ids=entity_position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=True)
        loss = None
        mlm_loss = None
        logits = self.lm_head(outputs.last_hidden_state)
        if labels is not None:
            labels = labels.to(logits.device)
            mlm_loss = self.loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))
            if loss is None:
                loss = mlm_loss
        mep_loss = None
        entity_logits = None
        if outputs.entity_last_hidden_state is not None:
            entity_logits = self.entity_predictions(outputs.entity_last_hidden_state)
            if entity_labels is not None:
                mep_loss = self.loss_fn(entity_logits.view(-1, self.config.entity_vocab_size), entity_labels.view(-1))
                if loss is None:
                    loss = mep_loss
                else:
                    loss = loss + mep_loss
        if not return_dict:
            return tuple((v for v in [loss, mlm_loss, mep_loss, logits, entity_logits, outputs.hidden_states, outputs.entity_hidden_states, outputs.attentions] if v is not None))
        return LukeMaskedLMOutput(loss=loss, mlm_loss=mlm_loss, mep_loss=mep_loss, logits=logits, entity_logits=entity_logits, hidden_states=outputs.hidden_states, entity_hidden_states=outputs.entity_hidden_states, attentions=outputs.attentions)