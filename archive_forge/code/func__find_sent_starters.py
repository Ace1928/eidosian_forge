import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def _find_sent_starters(self):
    """
        Uses collocation heuristics for each candidate token to
        determine if it frequently starts sentences.
        """
    for typ in self._sent_starter_fdist:
        if not typ:
            continue
        typ_at_break_count = self._sent_starter_fdist[typ]
        typ_count = self._type_fdist[typ] + self._type_fdist[typ + '.']
        if typ_count < typ_at_break_count:
            continue
        log_likelihood = self._col_log_likelihood(self._sentbreak_count, typ_count, typ_at_break_count, self._type_fdist.N())
        if log_likelihood >= self.SENT_STARTER and self._type_fdist.N() / self._sentbreak_count > typ_count / typ_at_break_count:
            yield (typ, log_likelihood)