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
def _count_n_grams(self, token_lst, n):
    n_grams = defaultdict(int)
    for n_gram in NGramIterator(token_lst, n):
        n_grams[n_gram] += 1
    return n_grams