import re
import warnings
import array
from enchant.errors import TokenizerNotFoundError
class _TokenFilter:
    """Private inner class implementing the tokenizer-wrapping logic.

        This might seem convoluted, but we're trying to create something
        akin to a meta-class - when Filter(tknzr) is called it must return
        a *callable* that can then be applied to a particular string to
        perform the tokenization.  Since we need to manage a lot of state
        during tokenization, returning a class is the best option.
        """
    _DOC_ERRORS = ['tknzr']

    def __init__(self, tokenizer, skip, split):
        self._skip = skip
        self._split = split
        self._tokenizer = tokenizer
        self._curtok = empty_tokenize()
        self._curword = ''
        self._curpos = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        while True:
            try:
                word, pos = next(self._curtok)
                return (word, pos + self._curpos)
            except StopIteration:
                word, pos = next(self._tokenizer)
                while self._skip(self._to_string(word)):
                    word, pos = next(self._tokenizer)
                self._curword = word
                self._curpos = pos
                self._curtok = self._split(word)

    def _to_string(self, word):
        if type(word) is array.array:
            if word.typecode == 'u':
                return word.tounicode()
            elif word.typecode == 'c':
                return word.tostring()
        return word

    def _get_offset(self):
        return self._tokenizer.offset

    def _set_offset(self, offset):
        msg = "changing a tokenizers 'offset' attribute is deprecated; use the 'set_offset' method"
        warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
        self.set_offset(offset)
    offset = property(_get_offset, _set_offset)

    def set_offset(self, val, replaced=False):
        old_offset = self._tokenizer.offset
        self._tokenizer.set_offset(val, replaced=replaced)
        keep_curtok = True
        curtok_offset = val - self._curpos
        if old_offset > val:
            keep_curtok = False
        if curtok_offset < 0:
            keep_curtok = False
        if curtok_offset >= len(self._curword):
            keep_curtok = False
        if keep_curtok and (not replaced):
            self._curtok.set_offset(curtok_offset)
        else:
            self._curtok = empty_tokenize()
            self._curword = ''
            self._curpos = 0