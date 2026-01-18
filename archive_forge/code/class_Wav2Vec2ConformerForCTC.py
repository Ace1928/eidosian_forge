import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_wav2vec2_conformer import Wav2Vec2ConformerConfig
@add_start_docstrings('Wav2Vec2Conformer Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).', WAV2VEC2_CONFORMER_START_DOCSTRING)
class Wav2Vec2ConformerForCTC(Wav2Vec2ConformerPreTrainedModel):

    def __init__(self, config, target_lang: Optional[str]=None):
        super().__init__(config)
        self.wav2vec2_conformer = Wav2Vec2ConformerModel(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.target_lang = target_lang
        if config.vocab_size is None:
            raise ValueError(f"You are trying to instantiate {self.__class__} with a configuration that does not define the vocabulary size of the language model head. Please instantiate the model as follows: `Wav2Vec2ConformerForCTC.from_pretrained(..., vocab_size=vocab_size)`. or define `vocab_size` of your model's configuration.")
        output_hidden_size = config.output_hidden_size if hasattr(config, 'add_adapter') and config.add_adapter else config.hidden_size
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)
        self.post_init()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2_conformer.feature_extractor._freeze_parameters()

    @add_start_docstrings_to_model_forward(WAV2VEC2_CONFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=CausalLMOutput, config_class=_CONFIG_FOR_DOC, expected_output=_CTC_EXPECTED_OUTPUT, expected_loss=_CTC_EXPECTED_LOSS)
    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: Optional[torch.Tensor]=None) -> Union[Tuple, CausalLMOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2_conformer(input_values, attention_mask=attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f'Label values must be <= vocab_size: {self.config.vocab_size}')
            attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(log_probs, flattened_targets, input_lengths, target_lengths, blank=self.config.pad_token_id, reduction=self.config.ctc_loss_reduction, zero_infinity=self.config.ctc_zero_infinity)
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)