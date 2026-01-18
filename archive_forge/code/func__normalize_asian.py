import re
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import (
@classmethod
def _normalize_asian(cls: Type['_TercomTokenizer'], sentence: str) -> str:
    """Split Chinese chars and Japanese kanji down to character level."""
    sentence = re.sub('([\\u4e00-\\u9fff\\u3400-\\u4dbf])', ' \\1 ', sentence)
    sentence = re.sub('([\\u31c0-\\u31ef\\u2e80-\\u2eff])', ' \\1 ', sentence)
    sentence = re.sub('([\\u3300-\\u33ff\\uf900-\\ufaff\\ufe30-\\ufe4f])', ' \\1 ', sentence)
    sentence = re.sub('([\\u3200-\\u3f22])', ' \\1 ', sentence)
    sentence = re.sub('(^|^[\\u3040-\\u309f])([\\u3040-\\u309f]+)(?=$|^[\\u3040-\\u309f])', '\\1 \\2 ', sentence)
    sentence = re.sub('(^|^[\\u30a0-\\u30ff])([\\u30a0-\\u30ff]+)(?=$|^[\\u30a0-\\u30ff])', '\\1 \\2 ', sentence)
    sentence = re.sub('(^|^[\\u31f0-\\u31ff])([\\u31f0-\\u31ff]+)(?=$|^[\\u31f0-\\u31ff])', '\\1 \\2 ', sentence)
    sentence = re.sub(cls._ASIAN_PUNCTUATION, ' \\1 ', sentence)
    return re.sub(cls._FULL_WIDTH_PUNCTUATION, ' \\1 ', sentence)