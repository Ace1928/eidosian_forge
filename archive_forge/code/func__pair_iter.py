import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def _pair_iter(iterator):
    """
    Yields pairs of tokens from the given iterator such that each input
    token will appear as the first element in a yielded tuple. The last
    pair will have None as its second element.
    """
    iterator = iter(iterator)
    try:
        prev = next(iterator)
    except StopIteration:
        return
    for el in iterator:
        yield (prev, el)
        prev = el
    yield (prev, None)