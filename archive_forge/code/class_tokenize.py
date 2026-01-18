import re
import warnings
import array
from enchant.errors import TokenizerNotFoundError
class tokenize:
    """Base class for all tokenizer objects.

    Each tokenizer must be an iterator and provide the 'offset'
    attribute as described in the documentation for this module.

    While tokenizers are in fact classes, they should be treated
    like functions, and so are named using lower_case rather than
    the CamelCase more traditional of class names.
    """
    _DOC_ERRORS = ['CamelCase']

    def __init__(self, text):
        self._text = text
        self._offset = 0

    def __next__(self):
        return self.next()

    def next(self):
        raise NotImplementedError()

    def __iter__(self):
        return self

    def set_offset(self, offset, replaced=False):
        self._offset = offset

    def _get_offset(self):
        return self._offset

    def _set_offset(self, offset):
        msg = "changing a tokenizers 'offset' attribute is deprecated; use the 'set_offset' method"
        warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
        self.set_offset(offset)
    offset = property(_get_offset, _set_offset)