from ...language import BaseDefaults, Language
from ..nb import SYNTAX_ITERATORS
from .punctuation import TOKENIZER_INFIXES, TOKENIZER_PREFIXES, TOKENIZER_SUFFIXES
from .tokenizer_exceptions import TOKENIZER_EXCEPTIONS
class NorwegianNynorsk(Language):
    lang = 'nn'
    Defaults = NorwegianNynorskDefaults