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
def _tokenize_13a(cls: Type['_SacreBLEUTokenizer'], line: str) -> str:
    """Tokenizes a line using a relatively minimal tokenization that is equivalent to mteval-v13a, used by WMT.

        Args:
            line: input sentence

        Return:
            tokenized sentence

        """
    line = line.replace('<skipped>', '')
    line = line.replace('-\n', '')
    line = line.replace('\n', ' ')
    if '&' in line:
        line = line.replace('&quot;', '"')
        line = line.replace('&amp;', '&')
        line = line.replace('&lt;', '<')
        line = line.replace('&gt;', '>')
    return cls._tokenize_regex(f' {line} ')