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
class _SQuAD(SQuAD):
    """Wrapper for deprecated import.

    >>> preds = [{"prediction_text": "1976", "id": "56e10a3be3433e1400422b22"}]
    >>> target = [{"answers": {"answer_start": [97], "text": ["1976"]}, "id": "56e10a3be3433e1400422b22"}]
    >>> squad = _SQuAD()
    >>> squad(preds, target)
    {'exact_match': tensor(100.), 'f1': tensor(100.)}

    """

    def __init__(self, **kwargs: Any) -> None:
        _deprecated_root_import_class('SQuAD', 'text')
        super().__init__(**kwargs)