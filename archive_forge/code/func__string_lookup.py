import sys
from collections import Counter
from collections.abc import Iterable
from functools import singledispatch
from itertools import chain
@_dispatched_lookup.register(str)
def _string_lookup(word, vocab):
    """Looks up one word in the vocabulary."""
    return word if word in vocab else vocab.unk_label