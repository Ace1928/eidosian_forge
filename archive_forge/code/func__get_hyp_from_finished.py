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
def _get_hyp_from_finished(self, hypothesis_tail):
    """
        Extract hypothesis ending with EOS at timestep with hyp_id.

        :param timestep:
            timestep with range up to len(self.outputs) - 1

        :param hyp_id:
            id with range up to beam_size - 1

        :return:
            hypothesis sequence
        """
    hyp_idx = []
    endback = hypothesis_tail.hypid
    for i in range(hypothesis_tail.timestep, -1, -1):
        hyp_idx.append(_HypothesisTail(timestep=i, hypid=endback, score=self.all_scores[i][endback], tokenid=self.outputs[i][endback]))
        endback = self.bookkeep[i - 1][endback]
    return hyp_idx