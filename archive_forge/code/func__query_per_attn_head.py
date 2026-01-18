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
def _query_per_attn_head(self, hidden_states):
    per_head_query_key = self.query_key.weight.reshape(self.num_attention_heads, self.attention_head_size, self.hidden_size).transpose(-2, -1)
    query_key_vectors = torch.einsum('balh,ahr->balr', hidden_states, per_head_query_key)
    return query_key_vectors