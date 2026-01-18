from typing import Callable, Optional
from thinc.api import Model
from ...language import BaseDefaults, Language
from ...pipeline import Lemmatizer
from .lex_attrs import LEX_ATTRS
from .punctuation import TOKENIZER_SUFFIXES
from .stop_words import STOP_WORDS
from .syntax_iterators import SYNTAX_ITERATORS
from .tokenizer_exceptions import TOKENIZER_EXCEPTIONS
class Persian(Language):
    lang = 'fa'
    Defaults = PersianDefaults