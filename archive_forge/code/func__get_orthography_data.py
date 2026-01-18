import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def _get_orthography_data(self, tokens):
    """
        Collect information about whether each token type occurs
        with different case patterns (i) overall, (ii) at
        sentence-initial positions, and (iii) at sentence-internal
        positions.
        """
    context = 'internal'
    tokens = list(tokens)
    for aug_tok in tokens:
        if aug_tok.parastart and context != 'unknown':
            context = 'initial'
        if aug_tok.linestart and context == 'internal':
            context = 'unknown'
        typ = aug_tok.type_no_sentperiod
        flag = _ORTHO_MAP.get((context, aug_tok.first_case), 0)
        if flag:
            self._params.add_ortho_context(typ, flag)
        if aug_tok.sentbreak:
            if not (aug_tok.is_number or aug_tok.is_initial):
                context = 'initial'
            else:
                context = 'unknown'
        elif aug_tok.ellipsis or aug_tok.abbr:
            context = 'unknown'
        else:
            context = 'internal'