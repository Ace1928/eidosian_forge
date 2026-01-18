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
def _gather_by_expansion(self, vectors, idxs, num_hashes):
    """
        expand dims of idxs and vectors for all hashes and gather
        """
    expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
    vectors = vectors.repeat(1, 1, num_hashes, 1)
    return torch.gather(vectors, 2, expanded_idxs)