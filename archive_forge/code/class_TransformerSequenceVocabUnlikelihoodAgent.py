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
class TransformerSequenceVocabUnlikelihoodAgent(SequenceVocabUnlikelihoodAgentTrait, TransformerGeneratorAgent):
    """
    Example usage:

    -t convai2 -m parlai_internal.projects.dontsaythat.agents:TransformerSequenceVocabUnlikelihoodAgent
    """
    pass