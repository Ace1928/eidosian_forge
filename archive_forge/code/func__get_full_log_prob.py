from collections import namedtuple
import torch
from torch import Tensor
from typing import List, Sequence
from . import Sequential, ModuleList, Linear
from .module import Module
from ..functional import log_softmax
def _get_full_log_prob(self, input, head_output):
    """Given input tensor, and output of ``self.head``, compute the log of the full distribution."""
    out = input.new_empty((head_output.size(0), self.n_classes))
    head_logprob = log_softmax(head_output, dim=1)
    out[:, :self.shortlist_size] = head_logprob[:, :self.shortlist_size]
    for i, (start_idx, stop_idx) in enumerate(zip(self.cutoffs, self.cutoffs[1:])):
        cluster_output = self.tail[i](input)
        cluster_logprob = log_softmax(cluster_output, dim=1)
        output_logprob = cluster_logprob + head_logprob[:, self.shortlist_size + i].unsqueeze(1)
        out[:, start_idx:stop_idx] = output_logprob
    return out