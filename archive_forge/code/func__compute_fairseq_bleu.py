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
def _compute_fairseq_bleu(self, batch: Batch, preds):
    """
        Compute BLEU score between text and label, using the FAIRSeq BLEU Scorer.

        :param batch:
            Batch of observations
        :param texts:
            list of string predictions
        """
    all_results = []
    label_vec = batch.label_vec
    assert label_vec is not None, 'label_vec must exist for fairseq bleu'
    for i, t in enumerate(preds):
        result = FairseqBleuMetric.compute_many(t[1:], label_vec[i].unsqueeze(0), pad_idx=self.NULL_IDX, end_idx=self.END_IDX, unk_idx=self.dict[self.dict.unk_token])
        if result is None:
            return
        all_results.append(result)
    bleu_scores = list(zip(*all_results))
    for k in range(4):
        self.record_local_metric(f'fairseq_bleu{k + 1}', bleu_scores[k])