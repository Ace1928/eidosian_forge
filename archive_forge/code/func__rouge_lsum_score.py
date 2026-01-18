import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.imports import _NLTK_AVAILABLE
def _rouge_lsum_score(pred: Sequence[Sequence[str]], target: Sequence[Sequence[str]]) -> Dict[str, Tensor]:
    """Compute precision, recall and F1 score for the Rouge-LSum metric.

    More information can be found in Section 3.2 of the referenced paper [1]. This implementation follow the official
    implementation from:
    https://github.com/google-research/google-research/blob/master/rouge/rouge_scorer.py.

    Args:
        pred: An iterable of predicted sentence split by '\\n'.
        target: An iterable target sentence split by '\\n'.

    References:
        [1] ROUGE: A Package for Automatic Evaluation of Summaries by Chin-Yew Lin. https://aclanthology.org/W04-1013/

    """
    pred_len = sum(map(len, pred))
    target_len = sum(map(len, target))
    if 0 in (pred_len, target_len):
        return {'precision': tensor(0.0), 'recall': tensor(0.0), 'fmeasure': tensor(0.0)}

    def _get_token_counts(sentences: Sequence[Sequence[str]]) -> Counter:
        ngrams: Counter = Counter()
        for sentence in sentences:
            ngrams.update(sentence)
        return ngrams
    pred_tokens_count = _get_token_counts(pred)
    target_tokens_count = _get_token_counts(target)
    hits = 0
    for tgt in target:
        lcs = _union_lcs(pred, tgt)
        for token in lcs:
            if pred_tokens_count[token] > 0 and target_tokens_count[token] > 0:
                hits += 1
                pred_tokens_count[token] -= 1
                target_tokens_count[token] -= 1
    return _compute_metrics(hits, pred_len, target_len)