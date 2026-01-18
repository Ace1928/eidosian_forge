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
def _tokenize_ja_mecab(cls: Type['_SacreBLEUTokenizer'], line: str) -> str:
    """Tokenizes a Japanese string line using MeCab morphological analyzer.

        Args:
            line: the input string to tokenize.

        Return:
            The tokenized string.

        """
    import ipadic
    import MeCab
    tagger = MeCab.Tagger(ipadic.MECAB_ARGS + ' -Owakati')
    line = line.strip()
    return tagger.parse(line).strip()