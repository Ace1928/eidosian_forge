import array
import warnings
import enchant
from enchant.errors import (
from enchant.tokenize import get_tokenizer
from enchant.utils import get_default_language
def _array_to_string(self, text):
    """Format an internal array as a standard string."""
    if text.typecode == 'u':
        return text.tounicode()
    return text.tostring()