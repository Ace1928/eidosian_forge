from typing import Any, Literal, Optional, Sequence
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text.cer import CharErrorRate
from torchmetrics.text.chrf import CHRFScore
from torchmetrics.text.eed import ExtendedEditDistance
from torchmetrics.text.mer import MatchErrorRate
from torchmetrics.text.perplexity import Perplexity
from torchmetrics.text.sacre_bleu import SacreBLEUScore
from torchmetrics.text.squad import SQuAD
from torchmetrics.text.ter import TranslationEditRate
from torchmetrics.text.wer import WordErrorRate
from torchmetrics.text.wil import WordInfoLost
from torchmetrics.text.wip import WordInfoPreserved
from torchmetrics.utilities.prints import _deprecated_root_import_class
class _BLEUScore(BLEUScore):
    """Wrapper for deprecated import.

    >>> preds = ['the cat is on the mat']
    >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
    >>> bleu = _BLEUScore()
    >>> bleu(preds, target)
    tensor(0.7598)

    """

    def __init__(self, n_gram: int=4, smooth: bool=False, weights: Optional[Sequence[float]]=None, **kwargs: Any) -> None:
        _deprecated_root_import_class('BLEUScore', 'text')
        super().__init__(n_gram=n_gram, smooth=smooth, weights=weights, **kwargs)