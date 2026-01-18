from ...language import BaseDefaults, Language
from .punctuation import TOKENIZER_INFIXES, TOKENIZER_PREFIXES, TOKENIZER_SUFFIXES
from .stop_words import STOP_WORDS
from .tokenizer_exceptions import TOKEN_MATCH, TOKENIZER_EXCEPTIONS
class Hungarian(Language):
    lang = 'hu'
    Defaults = HungarianDefaults