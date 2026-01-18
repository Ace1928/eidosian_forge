import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ....modeling_utils import PreTrainedModel
from ....utils import (
from .configuration_transfo_xl import TransfoXLConfig
from .modeling_transfo_xl_utilities import ProjectedAdaptiveLogSoftmax
@add_start_docstrings('\n    The Transformer-XL Model transformer with a sequence classification head on top (linear layer).\n\n    [`TransfoXLForSequenceClassification`] uses the last token in order to do the classification, as other causal\n    models (e.g. GPT-1) do.\n\n    Since it does classification on the last token, it requires to know the position of the last token. If a\n    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If\n    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the\n    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in\n    each row of the batch).\n    ', TRANSFO_XL_START_DOCSTRING)
class TransfoXLForSequenceClassification(TransfoXLPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = TransfoXLModel(config)
        self.score = nn.Linear(config.d_embed, self.num_labels, bias=False)
        self.post_init()

    @add_start_docstrings_to_model_forward(TRANSFO_XL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=TransfoXLSequenceClassifierOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, mems: Optional[List[torch.FloatTensor]]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, TransfoXLSequenceClassifierOutputWithPast]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(input_ids, mems=mems, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)
        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]
        assert self.config.pad_token_id is not None or batch_size == 1, 'Cannot handle batch sizes > 1 if no padding token is defined.'
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        elif input_ids is not None:
            sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(logits.device)
        else:
            sequence_lengths = -1
            logger.warning(f'{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be unexpected if using padding tokens in conjunction with `inputs_embeds.`')
        pooled_logits = logits[range(batch_size), sequence_lengths]
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = 'regression'
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = 'single_label_classification'
                else:
                    self.config.problem_type = 'multi_label_classification'
            if self.config.problem_type == 'regression':
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == 'multi_label_classification':
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return TransfoXLSequenceClassifierOutputWithPast(loss=loss, logits=pooled_logits, mems=transformer_outputs.mems, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)