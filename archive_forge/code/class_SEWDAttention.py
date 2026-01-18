import math
import warnings
from collections.abc import Sequence
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import softmax_backward_data
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_sew_d import SEWDConfig
class SEWDAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = DisentangledSelfAttention(config)
        self.output = SEWDSelfOutput(config)
        self.config = config

    def forward(self, hidden_states, attention_mask, output_attentions=False, query_states=None, relative_pos=None, rel_embeddings=None):
        self_output = self.self(hidden_states, attention_mask, output_attentions, query_states=query_states, relative_pos=relative_pos, rel_embeddings=rel_embeddings)
        if output_attentions:
            self_output, att_matrix = self_output
        if query_states is None:
            query_states = hidden_states
        attention_output = self.output(self_output, query_states)
        if output_attentions:
            return (attention_output, att_matrix)
        else:
            return attention_output