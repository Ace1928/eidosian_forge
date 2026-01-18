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
def _tokenize_flores(cls: Type['_SacreBLEUTokenizer'], line: str, tokenize: Literal['flores101', 'flores200']) -> str:
    """Tokenizes a string line using sentencepiece tokenizer.

        Args:
            line: the input string to tokenize.
            tokenize: Tokenization technique to be used.

        Return:
            The tokenized string.

        """
    import sentencepiece
    if cls.sentencepiece_processors[tokenize] is None:
        cls.sentencepiece_processors[tokenize] = sentencepiece.SentencePieceProcessor()
        file_path = os.path.join(_FLORES_LOCAL_DIR, _FLORES_MODELS_URL[tokenize].split('/')[-1])
        if not os.path.exists(file_path):
            cls.download_flores_file(tokenize)
        cls.sentencepiece_processors[tokenize].Load(file_path)
    return ' '.join(cls.sentencepiece_processors[tokenize].EncodeAsPieces(line))