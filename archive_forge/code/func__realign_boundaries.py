import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def _realign_boundaries(self, text: str, slices: Iterator[slice]) -> Iterator[slice]:
    """
        Attempts to realign punctuation that falls after the period but
        should otherwise be included in the same sentence.

        For example: "(Sent1.) Sent2." will otherwise be split as::

            ["(Sent1.", ") Sent1."].

        This method will produce::

            ["(Sent1.)", "Sent2."].
        """
    realign = 0
    for sentence1, sentence2 in _pair_iter(slices):
        sentence1 = slice(sentence1.start + realign, sentence1.stop)
        if not sentence2:
            if text[sentence1]:
                yield sentence1
            continue
        m = self._lang_vars.re_boundary_realignment.match(text[sentence2])
        if m:
            yield slice(sentence1.start, sentence2.start + len(m.group(0).rstrip()))
            realign = m.end()
        else:
            realign = 0
            if text[sentence1]:
                yield sentence1