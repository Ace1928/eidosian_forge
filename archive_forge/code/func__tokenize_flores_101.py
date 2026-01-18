import os
import re
import tempfile
from functools import partial
from typing import Any, ClassVar, Dict, Optional, Sequence, Type
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.functional.text.bleu import _bleu_score_compute, _bleu_score_update
from torchmetrics.utilities.imports import (
@classmethod
def _tokenize_flores_101(cls: Type['_SacreBLEUTokenizer'], line: str) -> str:
    """Tokenizes a string line using sentencepiece tokenizer according to `FLORES-101`_ dataset.

        Args:
            line: the input string to tokenize.

        Return:
            The tokenized string.

        """
    return cls._tokenize_flores(line, 'flores101')