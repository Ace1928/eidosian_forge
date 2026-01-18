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
def _tokenize_base(cls: Type['_SacreBLEUTokenizer'], line: str) -> str:
    """Tokenizes an input line with the tokenizer.

        Args:
            line: a segment to tokenize

        Return:
            the tokenized line

        """
    return line