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
class BeamSearch(TreeSearch):
    """
    Beam search.
    """

    def select_paths(self, logprobs, prior_scores, current_length):
        """
        Select the next vocabulary item in these beams.
        """
        if prior_scores.numel() == 1:
            logprobs = logprobs[0:1]
        beam_scores = logprobs + prior_scores.unsqueeze(1).expand_as(logprobs)
        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_idxs = torch.topk(flat_beam_scores, self.beam_size, dim=-1)
        voc_size = logprobs.size(-1)
        hyp_ids = best_idxs // voc_size
        tok_ids = best_idxs % voc_size
        return (hyp_ids, tok_ids, best_scores)