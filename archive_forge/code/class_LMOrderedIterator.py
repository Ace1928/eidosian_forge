import glob
import os
import pickle
import re
from collections import Counter, OrderedDict
from typing import List, Optional, Tuple
import numpy as np
from ....tokenization_utils import PreTrainedTokenizer
from ....utils import (
class LMOrderedIterator(object):

    def __init__(self, data, bsz, bptt, device='cpu', ext_len=None):
        """
        data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0
        self.device = device
        self.n_step = data.size(0) // bsz
        data = data.narrow(0, 0, self.n_step * bsz)
        self.data = data.view(bsz, -1).t().contiguous().to(device)
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

    def get_batch(self, i, bptt=None):
        if bptt is None:
            bptt = self.bptt
        seq_len = min(bptt, self.data.size(0) - 1 - i)
        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)
        data = self.data[beg_idx:end_idx]
        target = self.data[i + 1:i + 1 + seq_len]
        data_out = data.transpose(0, 1).contiguous().to(self.device)
        target_out = target.transpose(0, 1).contiguous().to(self.device)
        return (data_out, target_out, seq_len)

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.data.size(0) - 1, self.bptt):
            yield self.get_batch(i)

    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.0
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            data, target, seq_len = self.get_batch(i, bptt)
            i += seq_len
            yield (data, target, seq_len)
            if i >= self.data.size(0) - 2:
                break

    def __iter__(self):
        return self.get_fixlen_iter()