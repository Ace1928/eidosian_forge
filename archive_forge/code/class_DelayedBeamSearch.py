from abc import ABC, abstractmethod
from typing import TypeVar, List, Dict, Optional, Tuple, Set, Iterable
import math
from operator import attrgetter
import torch
import torch.nn as nn
import torch.nn.functional as F
from parlai.core.opt import Opt
from parlai.utils.distributed import is_distributed, sync_parameters
from parlai.core.torch_agent import TorchAgent, Batch, Output, DictionaryAgent
from parlai.utils.misc import warn_once
import parlai.utils.logging as logging
from parlai.core.metrics import (
from parlai.utils.fp16 import FP16SafeCrossEntropy
from parlai.utils.torch import (
class DelayedBeamSearch(TreeSearch):
    """
    DelayedBeam: Top-K sampling followed by beam search (Massarelli et al., 2019).

    Samples from a truncated distribution where only the most probable K words
    are considered at each time for the first N tokens, then switches to beam
    after N steps.

    See https://arxiv.org/abs/1911.03587 for details.
    """

    def __init__(self, k, delay, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.delay = delay

    def select_paths(self, logprobs, prior_scores, current_length):
        if current_length < self.delay:
            return TopKSampling.select_paths(self, logprobs, prior_scores, current_length)
        else:
            return BeamSearch.select_paths(self, logprobs, prior_scores, current_length)