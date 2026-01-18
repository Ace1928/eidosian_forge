import glob
import os
import pickle
import re
from collections import Counter, OrderedDict
from typing import List, Optional, Tuple
import numpy as np
from ....tokenization_utils import PreTrainedTokenizer
from ....utils import (
class LMMultiFileIterator(LMShuffledIterator):

    def __init__(self, paths, vocab, bsz, bptt, device='cpu', ext_len=None, shuffle=False):
        self.paths = paths
        self.vocab = vocab
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0
        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self, path):
        sents = self.vocab.encode_file(path, add_double_eos=True)
        if self.shuffle:
            np.random.shuffle(sents)
        sent_stream = iter(sents)
        return sent_stream

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.paths)
        for path in self.paths:
            sent_stream = self.get_sent_stream(path)
            for batch in self.stream_iterator(sent_stream):
                yield batch