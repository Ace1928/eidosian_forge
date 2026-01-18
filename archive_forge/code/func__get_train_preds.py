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
def _get_train_preds(self, scores, label_inds, cands, cand_vecs):
    """
        Return predictions from training.
        """
    batchsize = scores.size(0)
    if self.rank_top_k > 0:
        _, ranks = scores.topk(min(self.rank_top_k, scores.size(1)), 1, largest=True)
    else:
        _, ranks = scores.sort(1, descending=True)
    ranks_m = []
    mrrs_m = []
    for b in range(batchsize):
        rank = (ranks[b] == label_inds[b]).nonzero()
        rank = rank.item() if len(rank) == 1 else scores.size(1)
        ranks_m.append(1 + rank)
        mrrs_m.append(1.0 / (1 + rank))
    self.record_local_metric('rank', AverageMetric.many(ranks_m))
    self.record_local_metric('mrr', AverageMetric.many(mrrs_m))
    ranks = ranks.cpu()
    preds = []
    for i, ordering in enumerate(ranks):
        if cand_vecs.dim() == 2:
            cand_list = cands
        elif cand_vecs.dim() == 3:
            cand_list = cands[i]
        if len(ordering) != len(cand_list):
            for x in ordering:
                if x < len(cand_list):
                    preds.append(cand_list[x])
                    break
        else:
            preds.append(cand_list[ordering[0]])
    return Output(preds)