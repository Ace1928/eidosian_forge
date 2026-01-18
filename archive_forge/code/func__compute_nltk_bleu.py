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
def _compute_nltk_bleu(self, batch: Batch, texts: List[str]):
    """
        Compute BLEU score between text and label(s), using the NLTK BLEU Scorer.

        Note this differs from BLEU in ParlAI metrics in that the answers
        are unnormalized (no removal of stop words, etc.)

        :param batch:
            Batch of observations
        :param texts:
            list of string predictions
        """
    results: Dict[int, List[Metric]] = {}
    observations = batch.observations
    assert observations is not None, 'observations must not be none in nltk bleu'
    for i, p in enumerate(texts):
        obs = observations[i]
        references = []
        for lbl in obs['eval_labels']:
            references.append(self._v2t(self._vectorize_text(lbl, True, True, self.label_truncate, False)))
        for k in range(1, 5):
            b = BleuMetric.compute(p, references, k)
            if b is None:
                b = BleuMetric(0)
            if k not in results:
                results[k] = []
            results[k].append(b)
    for k in range(1, 5):
        self.record_local_metric(f'nltk_bleu{k}', results[k])