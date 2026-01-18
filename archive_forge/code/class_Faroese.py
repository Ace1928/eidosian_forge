from ...language import BaseDefaults, Language
from ..punctuation import TOKENIZER_INFIXES, TOKENIZER_PREFIXES, TOKENIZER_SUFFIXES
from .tokenizer_exceptions import TOKENIZER_EXCEPTIONS
class Faroese(Language):
    lang = 'fo'
    Defaults = FaroeseDefaults