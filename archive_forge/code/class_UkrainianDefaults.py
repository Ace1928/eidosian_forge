from typing import Callable, Optional
from thinc.api import Model
from ...language import BaseDefaults, Language
from ..punctuation import (
from .lemmatizer import UkrainianLemmatizer
from .lex_attrs import LEX_ATTRS
from .stop_words import STOP_WORDS
from .tokenizer_exceptions import TOKENIZER_EXCEPTIONS
class UkrainianDefaults(BaseDefaults):
    tokenizer_exceptions = TOKENIZER_EXCEPTIONS
    lex_attr_getters = LEX_ATTRS
    stop_words = STOP_WORDS
    suffixes = COMBINING_DIACRITICS_TOKENIZER_SUFFIXES
    infixes = COMBINING_DIACRITICS_TOKENIZER_INFIXES