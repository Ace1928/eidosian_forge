from ...language import BaseDefaults, Language
from .lex_attrs import LEX_ATTRS
from .stop_words import STOP_WORDS
class Czech(Language):
    lang = 'cs'
    Defaults = CzechDefaults