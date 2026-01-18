from typing import Dict, Any
from abc import abstractmethod
from itertools import islice
import os
from tqdm import tqdm
import random
import torch
from parlai.core.opt import Opt
from parlai.utils.distributed import is_distributed
from parlai.core.torch_agent import TorchAgent, Output
from parlai.utils.misc import warn_once
from parlai.utils.torch import (
from parlai.utils.fp16 import FP16SafeCrossEntropy
from parlai.core.metrics import AverageMetric
import parlai.utils.logging as logging
def _get_batch_train_metrics(self, scores):
    """
        Get fast metrics calculations if we train with batch candidates.

        Specifically, calculate accuracy ('train_accuracy'), average rank, and mean
        reciprocal rank.
        """
    batchsize = scores.size(0)
    targets = scores.new_empty(batchsize).long()
    targets = torch.arange(batchsize, out=targets)
    nb_ok = (scores.max(dim=1)[1] == targets).float()
    self.record_local_metric('train_accuracy', AverageMetric.many(nb_ok))
    above_dot_prods = scores - scores.diag().view(-1, 1)
    ranks = (above_dot_prods > 0).float().sum(dim=1) + 1
    mrr = 1.0 / (ranks + 1e-05)
    self.record_local_metric('rank', AverageMetric.many(ranks))
    self.record_local_metric('mrr', AverageMetric.many(mrr))