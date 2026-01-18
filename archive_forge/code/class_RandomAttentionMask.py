import gc
import math
from collections import namedtuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch
import triton
from triton.ops.blocksparse import matmul as blocksparse_matmul
from xformers.benchmarks.utils import pretty_barplot
from xformers.components.attention.attention_patterns import (
from xformers.components.attention.core import SparseCS, _matmul_with_mask
class RandomAttentionMask(AttentionMask):
    """
    This is a Random mask. Useful for performance and memory analysis.
    """

    def __init__(self, config=None):
        super(RandomAttentionMask, self).__init__(config)
        self.set_config_attr('mask_prob', 0.5)

    def gen_mask(self, keep_blocked=True):
        seq_length = self.config.seq_length
        if keep_blocked:
            seq_length = self.config.blocked_seq_length
        mask = torch.rand(seq_length, seq_length) > self.config.mask_prob
        return self.expand(mask)

    def __str__(self):
        return 'random'