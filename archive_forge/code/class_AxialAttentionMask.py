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
class AxialAttentionMask(AttentionMask):
    """
    BigBird mask are composed of three types of masks - random, global and window.
    For more details, refer to https://arxiv.org/pdf/2007.14062.pdf

    One point to note is that mask is per head here. So, mask is 3D tensor.
    (num_heads, seq_length, seq_length).
    """

    def __init__(self, config=None):
        super(AxialAttentionMask, self).__init__(config)
        if config is None:
            self.set_config_attr('seq_length', 1024)

    def is_valid_config(self, keep_blocked=True):
        seq_length = self.config.seq_length
        if keep_blocked:
            seq_length = self.config.blocked_seq_length
        H = int(math.sqrt(seq_length))
        if H * H == seq_length:
            return True
        return False

    def gen_mask(self, keep_blocked=True):
        seq_length = self.config.seq_length
        if keep_blocked:
            seq_length = self.config.blocked_seq_length
        H = int(math.sqrt(seq_length))
        assert H * H == seq_length, f'H={H}, seq_length={seq_length}'
        return self.expand(axial_2d_pattern(H, H))

    def __str__(self):
        return 'axial'