import re
import warnings
import array
from enchant.errors import TokenizerNotFoundError
class empty_tokenize(tokenize):
    """Tokenizer class that yields no elements."""
    _DOC_ERRORS = []

    def __init__(self):
        super().__init__('')

    def next(self):
        raise StopIteration()