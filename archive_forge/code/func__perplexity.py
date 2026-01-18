import os
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics.functional.text.bert import bert_score
from torchmetrics.functional.text.bleu import bleu_score
from torchmetrics.functional.text.cer import char_error_rate
from torchmetrics.functional.text.chrf import chrf_score
from torchmetrics.functional.text.eed import extended_edit_distance
from torchmetrics.functional.text.infolm import (
from torchmetrics.functional.text.infolm import infolm
from torchmetrics.functional.text.mer import match_error_rate
from torchmetrics.functional.text.perplexity import perplexity
from torchmetrics.functional.text.rouge import rouge_score
from torchmetrics.functional.text.sacre_bleu import sacre_bleu_score
from torchmetrics.functional.text.squad import squad
from torchmetrics.functional.text.ter import translation_edit_rate
from torchmetrics.functional.text.wer import word_error_rate
from torchmetrics.functional.text.wil import word_information_lost
from torchmetrics.functional.text.wip import word_information_preserved
from torchmetrics.utilities.imports import _TRANSFORMERS_GREATER_EQUAL_4_4
from torchmetrics.utilities.prints import _deprecated_root_import_func
def _perplexity(preds: Tensor, target: Tensor, ignore_index: Optional[int]=None) -> Tensor:
    """Wrapper for deprecated import.

    >>> import torch
    >>> gen = torch.manual_seed(42)
    >>> preds = torch.rand(2, 8, 5, generator=gen)
    >>> target = torch.randint(5, (2, 8), generator=gen)
    >>> target[0, 6:] = -100
    >>> _perplexity(preds, target, ignore_index=-100)
    tensor(5.8540)

    """
    _deprecated_root_import_func('perplexity', 'text')
    return perplexity(preds=preds, target=target, ignore_index=ignore_index)