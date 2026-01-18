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
def _construct_token_losses(self, labels, model_output):
    scores, _, _ = model_output
    score_view = scores.view(-1, scores.size(-1))
    losses = self.criterion(score_view, labels.view(-1)).view(len(labels), -1)
    token_losses = []
    for i, label in enumerate(labels):
        token_losses.append(list(zip([self.dict[token] for token in label.tolist()], losses[i].tolist())))
    return token_losses