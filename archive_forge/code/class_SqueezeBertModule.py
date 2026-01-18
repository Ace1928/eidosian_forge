import math
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_squeezebert import SqueezeBertConfig
class SqueezeBertModule(nn.Module):

    def __init__(self, config):
        """
        - hidden_size = input chans = output chans for Q, K, V (they are all the same ... for now) = output chans for
          the module
        - intermediate_size = output chans for intermediate layer
        - groups = number of groups for all layers in the BertModule. (eventually we could change the interface to
          allow different groups for different layers)
        """
        super().__init__()
        c0 = config.hidden_size
        c1 = config.hidden_size
        c2 = config.intermediate_size
        c3 = config.hidden_size
        self.attention = SqueezeBertSelfAttention(config=config, cin=c0, q_groups=config.q_groups, k_groups=config.k_groups, v_groups=config.v_groups)
        self.post_attention = ConvDropoutLayerNorm(cin=c0, cout=c1, groups=config.post_attention_groups, dropout_prob=config.hidden_dropout_prob)
        self.intermediate = ConvActivation(cin=c1, cout=c2, groups=config.intermediate_groups, act=config.hidden_act)
        self.output = ConvDropoutLayerNorm(cin=c2, cout=c3, groups=config.output_groups, dropout_prob=config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask, output_attentions):
        att = self.attention(hidden_states, attention_mask, output_attentions)
        attention_output = att['context_layer']
        post_attention_output = self.post_attention(attention_output, hidden_states)
        intermediate_output = self.intermediate(post_attention_output)
        layer_output = self.output(intermediate_output, post_attention_output)
        output_dict = {'feature_map': layer_output}
        if output_attentions:
            output_dict['attention_score'] = att['attention_score']
        return output_dict