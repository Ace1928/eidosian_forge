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
def _set_num_buckets(self, sequence_length):
    num_buckets_pow_2 = (2 * (sequence_length // self.chunk_length)).bit_length() - 1
    num_buckets = 2 ** num_buckets_pow_2
    num_buckets_limit = 2 * max(int((self.max_position_embeddings // self.chunk_length) ** 0.5), self.chunk_length)
    if num_buckets > num_buckets_limit:
        num_buckets = [2 ** (num_buckets_pow_2 // 2), 2 ** (num_buckets_pow_2 - num_buckets_pow_2 // 2)]
    logger.warning(f'config.num_buckets is not set. Setting config.num_buckets to {num_buckets}...')
    self.config.num_buckets = num_buckets
    self.num_buckets = num_buckets