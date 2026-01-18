import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as spm
from ...tokenization_utils import AddedToken, BatchEncoding, PreTrainedTokenizer
from ...utils import logging
def _convert_lang_code_special_format(self, lang: str) -> str:
    """Convert Language Codes to format tokenizer uses if required"""
    lang = FAIRSEQ_LANGUAGE_CODES_MAP[lang] if lang in FAIRSEQ_LANGUAGE_CODES_MAP.keys() else lang
    return lang