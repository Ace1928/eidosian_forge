import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.imports import _NLTK_AVAILABLE
def _rouge_score_update(preds: Sequence[str], target: Sequence[Sequence[str]], rouge_keys_values: List[Union[int, str]], accumulate: str, stemmer: Optional[Any]=None, normalizer: Optional[Callable[[str], str]]=None, tokenizer: Optional[Callable[[str], Sequence[str]]]=None) -> Dict[Union[int, str], List[Dict[str, Tensor]]]:
    """Update the rouge score with the current set of predicted and target sentences.

    Args:
        preds: An iterable of predicted sentences.
        target: An iterable of iterable of target sentences.
        rouge_keys_values: List of N-grams/'L'/'Lsum' arguments.
        accumulate: Useful in case of multi-reference rouge score.
            ``avg`` takes the avg of all references with respect to predictions
            ``best`` takes the best fmeasure score obtained between prediction and multiple corresponding references.
            Allowed values are ``avg`` and ``best``.
        stemmer: Porter stemmer instance to strip word suffixes to improve matching.
        normalizer:
            A user's own normalizer function.
            If this is ``None``, replacing any non-alpha-numeric characters with spaces is default.
            This function must take a `str` and return a `str`.
        tokenizer:
            A user's own tokenizer function. If this is ``None``, splitting by spaces is default
            This function must take a `str` and return `Sequence[str]`

    Example:
        >>> preds = "My name is John".split()
        >>> target = "Is your name John".split()
        >>> from pprint import pprint
        >>> score = _rouge_score_update(preds, target, rouge_keys_values=[1, 2, 3, 'L'], accumulate='best')
        >>> pprint(score)
        {1: [{'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)},
             {'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)},
             {'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)},
             {'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)}],
         2: [{'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)},
             {'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)},
             {'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)},
             {'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)}],
         3: [{'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)},
             {'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)},
             {'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)},
             {'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)}],
         'L': [{'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)},
               {'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)},
               {'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)},
               {'fmeasure': tensor(0.), 'precision': tensor(0.), 'recall': tensor(0.)}]}

    """
    results: Dict[Union[int, str], List[Dict[str, Tensor]]] = {rouge_key: [] for rouge_key in rouge_keys_values}
    for pred_raw, target_raw in zip(preds, target):
        result_inner: Dict[Union[int, str], Dict[str, Tensor]] = {rouge_key: {} for rouge_key in rouge_keys_values}
        result_avg: Dict[Union[int, str], List[Dict[str, Tensor]]] = {rouge_key: [] for rouge_key in rouge_keys_values}
        list_results = []
        pred = _normalize_and_tokenize_text(pred_raw, stemmer, normalizer, tokenizer)
        if 'Lsum' in rouge_keys_values:
            pred_lsum = [_normalize_and_tokenize_text(pred_sentence, stemmer, normalizer, tokenizer) for pred_sentence in _split_sentence(pred_raw)]
        for target_raw_inner in target_raw:
            tgt = _normalize_and_tokenize_text(target_raw_inner, stemmer, normalizer, tokenizer)
            if 'Lsum' in rouge_keys_values:
                target_lsum = [_normalize_and_tokenize_text(tgt_sentence, stemmer, normalizer, tokenizer) for tgt_sentence in _split_sentence(target_raw_inner)]
            for rouge_key in rouge_keys_values:
                if isinstance(rouge_key, int):
                    score = _rouge_n_score(pred, tgt, rouge_key)
                elif rouge_key == 'L':
                    score = _rouge_l_score(pred, tgt)
                elif rouge_key == 'Lsum':
                    score = _rouge_lsum_score(pred_lsum, target_lsum)
                result_inner[rouge_key] = score
                result_avg[rouge_key].append(score)
            list_results.append(result_inner.copy())
        if accumulate == 'best':
            key_curr = rouge_keys_values[0]
            all_fmeasure = torch.tensor([v[key_curr]['fmeasure'] for v in list_results])
            highest_idx = int(torch.argmax(all_fmeasure).item())
            for rouge_key in rouge_keys_values:
                results[rouge_key].append(list_results[highest_idx][rouge_key])
        elif accumulate == 'avg':
            new_result_avg: Dict[Union[int, str], Dict[str, Tensor]] = {rouge_key: {} for rouge_key in rouge_keys_values}
            for rouge_key, metrics in result_avg.items():
                _dict_metric_score_batch: Dict[str, List[Tensor]] = {}
                for metric in metrics:
                    for _type, value in metric.items():
                        if _type not in _dict_metric_score_batch:
                            _dict_metric_score_batch[_type] = []
                        _dict_metric_score_batch[_type].append(value)
                new_result_avg[rouge_key] = {_type: torch.tensor(_dict_metric_score_batch[_type]).mean() for _type in _dict_metric_score_batch}
            for rouge_key in rouge_keys_values:
                results[rouge_key].append(new_result_avg[rouge_key])
    return results