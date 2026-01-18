import sys
from collections import namedtuple
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.autograd.function import Function
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import CausalLMOutput, MaskedLMOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_reformer import ReformerConfig
class ReformerLayer(nn.Module):

    def __init__(self, config, layer_id=0):
        super().__init__()
        self.attention = ReformerAttention(config, layer_id)
        self.attention_seed = None
        self.feed_forward_seed = None
        self.feed_forward = ChunkReformerFeedForward(config)

    def _init_attention_seed(self):
        """
        This function sets a new seed for the attention layer to make dropout deterministic for both forward calls: 1
        normal forward call and 1 forward call in backward to recalculate activations.
        """
        if hasattr(torch.cuda, 'default_generators') and len(torch.cuda.default_generators) > 0:
            device_idx = torch.cuda.current_device()
            self.attention_seed = torch.cuda.default_generators[device_idx].seed()
        else:
            self.attention_seed = int(torch.seed() % sys.maxsize)
        torch.manual_seed(self.attention_seed)

    def _init_feed_forward_seed(self):
        """
        This function sets a new seed for the feed forward layer to make dropout deterministic for both forward calls:
        1 normal forward call and 1 forward call in backward to recalculate activations.
        """
        if hasattr(torch.cuda, 'default_generators') and len(torch.cuda.default_generators) > 0:
            device_idx = torch.cuda.current_device()
            self.feed_forward_seed = torch.cuda.default_generators[device_idx].seed()
        else:
            self.feed_forward_seed = int(torch.seed() % sys.maxsize)
        torch.manual_seed(self.feed_forward_seed)

    def forward(self, prev_attn_output, hidden_states, attention_mask=None, head_mask=None, num_hashes=None, past_buckets_states=None, use_cache=False, orig_sequence_length=None, output_attentions=False):
        with torch.no_grad():
            if self.training:
                self._init_attention_seed()
            attn_outputs = self.attention(hidden_states=hidden_states, head_mask=head_mask, attention_mask=attention_mask, num_hashes=num_hashes, past_buckets_states=past_buckets_states, use_cache=use_cache, orig_sequence_length=orig_sequence_length, output_attentions=output_attentions)
            attn_output = attn_outputs.hidden_states
            attn_output = prev_attn_output + attn_output
            del prev_attn_output
            if self.training:
                self._init_feed_forward_seed()
            hidden_states = hidden_states + self.feed_forward(attn_output)
        return ReformerOutput(attn_output=attn_output, hidden_states=hidden_states, attention_probs=attn_outputs.attention_probs, buckets=attn_outputs.buckets)

    def backward_pass(self, next_attn_output, hidden_states, grad_attn_output, grad_hidden_states, attention_mask=None, head_mask=None, buckets=None):
        assert self.training, 'If you want to train `ReformerModel` and its variations, make sure to use `model.train()` to put the model into training mode.'
        with torch.enable_grad():
            next_attn_output.requires_grad = True
            torch.manual_seed(self.feed_forward_seed)
            res_hidden_states = self.feed_forward(next_attn_output)
            res_hidden_states.backward(grad_hidden_states, retain_graph=True)
        with torch.no_grad():
            hidden_states = hidden_states - res_hidden_states
            del res_hidden_states
            grad_attn_output = grad_attn_output + next_attn_output.grad
            next_attn_output.grad = None
        with torch.enable_grad():
            hidden_states.requires_grad = True
            torch.manual_seed(self.attention_seed)
            output = self.attention(hidden_states=hidden_states, head_mask=head_mask, attention_mask=attention_mask, buckets=buckets).hidden_states
            output.backward(grad_attn_output, retain_graph=True)
        with torch.no_grad():
            attn_output = next_attn_output - output
            del output, next_attn_output
            grad_hidden_states = grad_hidden_states + hidden_states.grad
            hidden_states.grad = None
            hidden_states = hidden_states.detach()
        return ReformerBackwardOutput(attn_output=attn_output, hidden_states=hidden_states, grad_attn_output=grad_attn_output, grad_hidden_states=grad_hidden_states)