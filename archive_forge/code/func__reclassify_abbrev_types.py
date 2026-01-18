import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def _reclassify_abbrev_types(self, types):
    """
        (Re)classifies each given token if
          - it is period-final and not a known abbreviation; or
          - it is not period-final and is otherwise a known abbreviation
        by checking whether its previous classification still holds according
        to the heuristics of section 3.
        Yields triples (abbr, score, is_add) where abbr is the type in question,
        score is its log-likelihood with penalties applied, and is_add specifies
        whether the present type is a candidate for inclusion or exclusion as an
        abbreviation, such that:
          - (is_add and score >= 0.3)    suggests a new abbreviation; and
          - (not is_add and score < 0.3) suggests excluding an abbreviation.
        """
    for typ in types:
        if not _re_non_punct.search(typ) or typ == '##number##':
            continue
        if typ.endswith('.'):
            if typ in self._params.abbrev_types:
                continue
            typ = typ[:-1]
            is_add = True
        else:
            if typ not in self._params.abbrev_types:
                continue
            is_add = False
        num_periods = typ.count('.') + 1
        num_nonperiods = len(typ) - num_periods + 1
        count_with_period = self._type_fdist[typ + '.']
        count_without_period = self._type_fdist[typ]
        log_likelihood = self._dunning_log_likelihood(count_with_period + count_without_period, self._num_period_toks, count_with_period, self._type_fdist.N())
        f_length = math.exp(-num_nonperiods)
        f_periods = num_periods
        f_penalty = int(self.IGNORE_ABBREV_PENALTY) or math.pow(num_nonperiods, -count_without_period)
        score = log_likelihood * f_length * f_periods * f_penalty
        yield (typ, score, is_add)