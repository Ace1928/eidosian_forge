import re
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import (
def _ter_update(preds: Union[str, Sequence[str]], target: Sequence[Union[str, Sequence[str]]], tokenizer: _TercomTokenizer, total_num_edits: Tensor, total_tgt_length: Tensor, sentence_ter: Optional[List[Tensor]]=None) -> Tuple[Tensor, Tensor, Optional[List[Tensor]]]:
    """Update TER statistics.

    Args:
        preds: An iterable of hypothesis corpus.
        target: An iterable of iterables of reference corpus.
        tokenizer: An instance of ``_TercomTokenizer`` handling a sentence tokenization.
        total_num_edits: A total number of required edits to match hypothesis and reference sentences.
        total_tgt_length: A total average length of reference sentences.
        sentence_ter: A list of sentence-level TER values

    Return:
        total_num_edits:
            A total number of required edits to match hypothesis and reference sentences.
        total_tgt_length:
            A total average length of reference sentences.
        sentence_ter:
            (Optionally) A list of sentence-level TER.

    Raises:
        ValueError:
            If length of ``preds`` and ``target`` differs.

    """
    target, preds = _validate_inputs(target, preds)
    for pred, tgt in zip(preds, target):
        tgt_words_: List[List[str]] = [_preprocess_sentence(_tgt, tokenizer).split() for _tgt in tgt]
        pred_words_: List[str] = _preprocess_sentence(pred, tokenizer).split()
        num_edits, tgt_length = _compute_sentence_statistics(pred_words_, tgt_words_)
        total_num_edits += num_edits
        total_tgt_length += tgt_length
        if sentence_ter is not None:
            sentence_ter.append(_compute_ter_score_from_statistics(num_edits, tgt_length).unsqueeze(0))
    return (total_num_edits, total_tgt_length, sentence_ter)