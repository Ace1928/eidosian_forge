from typing import Callable, Optional
from thinc.api import Model
from ...language import BaseDefaults, Language
from ...pipeline import Lemmatizer
from .punctuation import TOKENIZER_INFIXES, TOKENIZER_PREFIXES, TOKENIZER_SUFFIXES
from .stop_words import STOP_WORDS
from .tokenizer_exceptions import TOKENIZER_EXCEPTIONS
class Bengali(Language):
    lang = 'bn'
    Defaults = BengaliDefaults