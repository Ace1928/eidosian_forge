from __future__ import absolute_import, division, print_function, unicode_literals
import codecs
from collections import defaultdict
from math import ceil, log as logf
import logging; log = logging.getLogger(__name__)
import pkg_resources
import os
from passlib import exc
from passlib.utils.compat import PY2, irange, itervalues, int_types
from passlib.utils import rng, getrandstr, to_unicode
from passlib.utils.decor import memoized_property
class PhraseGenerator(SequenceGenerator):
    """class which generates passphrases by randomly choosing
    from a list of unique words.

    :param wordset:
        wordset to draw from.
    :param preset:
        name of preset wordlist to use instead of ``wordset``.
    :param spaces:
        whether to insert spaces between words in output (defaults to ``True``).
    :param \\*\\*kwds:
        all other keywords passed to the :class:`SequenceGenerator` parent class.

    .. autoattribute:: wordset
    """
    wordset = 'eff_long'
    words = None
    sep = ' '

    def __init__(self, wordset=None, words=None, sep=None, **kwds):
        if words is not None:
            if wordset is not None:
                raise TypeError('`words` and `wordset` are mutually exclusive')
        else:
            if wordset is None:
                wordset = self.wordset
                assert wordset
            words = default_wordsets[wordset]
        self.wordset = wordset
        if not isinstance(words, _sequence_types):
            words = tuple(words)
        _ensure_unique(words, param='words')
        self.words = words
        if sep is None:
            sep = self.sep
        sep = to_unicode(sep, param='sep')
        self.sep = sep
        super(PhraseGenerator, self).__init__(**kwds)

    @memoized_property
    def symbol_count(self):
        return len(self.words)

    def __next__(self):
        words = (self.rng.choice(self.words) for _ in irange(self.length))
        return self.sep.join(words)