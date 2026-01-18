import sys
from collections import Counter
from collections.abc import Iterable
from functools import singledispatch
from itertools import chain
@singledispatch
def _dispatched_lookup(words, vocab):
    raise TypeError(f'Unsupported type for looking up in vocabulary: {type(words)}')