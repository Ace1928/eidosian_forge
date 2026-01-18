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
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
from .configuration_wav2vec2 import Wav2Vec2Config
@add_start_docstrings('Wav2Vec2 Model with a `language modeling` head on top.', WAV_2_VEC_2_START_DOCSTRING)
class Wav2Vec2ForMaskedLM(Wav2Vec2PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        warnings.warn('The class `Wav2Vec2ForMaskedLM` is deprecated. Please use `Wav2Vec2ForCTC` instead.', FutureWarning)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.post_init()

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    def forward(self, input_values: torch.FloatTensor, attention_mask: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: Optional[torch.Tensor]=None) -> Union[Tuple, MaskedLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(input_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return output
        return MaskedLMOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)