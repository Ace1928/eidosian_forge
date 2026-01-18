from ...language import BaseDefaults, Language
from .lex_attrs import LEX_ATTRS
from .stop_words import STOP_WORDS
from .syntax_iterators import SYNTAX_ITERATORS
from .tokenizer_exceptions import TOKEN_MATCH, TOKENIZER_EXCEPTIONS
class Turkish(Language):
    lang = 'tr'
    Defaults = TurkishDefaults