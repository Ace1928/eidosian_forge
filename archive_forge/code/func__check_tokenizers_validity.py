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
def _check_tokenizers_validity(cls: Type['_SacreBLEUTokenizer'], tokenize: _TokenizersLiteral) -> None:
    """Check if a supported tokenizer is chosen.

        Also check all dependencies of a given tokenizers are installed.

        """
    if tokenize not in cls._TOKENIZE_FN:
        raise ValueError(f'Unsupported tokenizer selected. Please, choose one of {list(cls._TOKENIZE_FN.keys())}')
    if tokenize == 'intl' and (not _REGEX_AVAILABLE):
        raise ModuleNotFoundError("`'intl'` tokenization requires that `regex` is installed. Use `pip install regex` or `pip install torchmetrics[text]`.")
    if tokenize == 'ja-mecab' and (not (_MECAB_AVAILABLE and _IPADIC_AVAILABLE)):
        raise ModuleNotFoundError("`'ja-mecab'` tokenization requires that `MeCab` and `ipadic` are installed. Use `pip install mecab-python3 ipadic` or `pip install torchmetrics[text]`.")
    if tokenize == 'ko-mecab' and (not (_MECAB_KO_AVAILABLE and _MECAB_KO_DIC_AVAILABLE)):
        raise ModuleNotFoundError("`'ko-mecab'` tokenization requires that `mecab_ko` and `mecab_ko_dic` are installed. Use `pip install mecab_ko mecab_ko_dic` or `pip install torchmetrics[text]`.")
    if 'flores' in tokenize and (not _SENTENCEPIECE_AVAILABLE):
        raise ModuleNotFoundError("`'flores101' and 'flores200'` tokenizations require that `sentencepiece` is installed. Use `pip install sentencepiece` or `pip install torchmetrics[text]`.")