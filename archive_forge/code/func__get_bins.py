import json
import math
import os
import torch
import torch.nn.functional as F
from collections import defaultdict, Counter
from nltk import ngrams
from parlai.agents.image_seq2seq.image_seq2seq import ImageSeq2seqAgent
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.metrics import AverageMetric, SumMetric, GlobalAverageMetric
from parlai.utils.misc import round_sigfigs
def _get_bins(self, counts: Counter):
    c = Counter()
    for k, v in counts.items():
        c.update({self.truebins.get(k, 'never'): v})
    t = sum(c.values())
    return {k: round_sigfigs(v / t, 4) for k, v in c.items()}