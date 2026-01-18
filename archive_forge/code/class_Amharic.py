from ...attrs import LANG
from ...language import BaseDefaults, Language
from ...util import update_exc
from ..tokenizer_exceptions import BASE_EXCEPTIONS
from .lex_attrs import LEX_ATTRS
from .punctuation import TOKENIZER_SUFFIXES
from .stop_words import STOP_WORDS
from .tokenizer_exceptions import TOKENIZER_EXCEPTIONS
class Amharic(Language):
    lang = 'am'
    Defaults = AmharicDefaults