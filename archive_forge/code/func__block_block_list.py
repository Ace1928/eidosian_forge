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
def _block_block_list(self, logprobs: torch.Tensor) -> torch.Tensor:
    if self.block_list is None:
        return logprobs
    for beam_id, hyp in enumerate(self.partial_hyps):
        for ngram_size, bad_ngrams in self.block_list.items():
            prefix = hyp[-(ngram_size - 1):]
            for ngram in bad_ngrams:
                if ngram_size == 1 or prefix == list(ngram[:-1]):
                    logprobs[beam_id][ngram[-1]] = neginf(logprobs.dtype)
    return logprobs