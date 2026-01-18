import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def _find_collocations(self):
    """
        Generates likely collocations and their log-likelihood.
        """
    for types in self._collocation_fdist:
        try:
            typ1, typ2 = types
        except TypeError:
            continue
        if typ2 in self._params.sent_starters:
            continue
        col_count = self._collocation_fdist[types]
        typ1_count = self._type_fdist[typ1] + self._type_fdist[typ1 + '.']
        typ2_count = self._type_fdist[typ2] + self._type_fdist[typ2 + '.']
        if typ1_count > 1 and typ2_count > 1 and (self.MIN_COLLOC_FREQ < col_count <= min(typ1_count, typ2_count)):
            log_likelihood = self._col_log_likelihood(typ1_count, typ2_count, col_count, self._type_fdist.N())
            if log_likelihood >= self.COLLOCATION and self._type_fdist.N() / typ1_count > typ2_count / col_count:
                yield ((typ1, typ2), log_likelihood)