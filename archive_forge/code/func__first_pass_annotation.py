import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def _first_pass_annotation(self, aug_tok: PunktToken) -> None:
    """
        Performs type-based annotation on a single token.
        """
    tok = aug_tok.tok
    if tok in self._lang_vars.sent_end_chars:
        aug_tok.sentbreak = True
    elif aug_tok.is_ellipsis:
        aug_tok.ellipsis = True
    elif aug_tok.period_final and (not tok.endswith('..')):
        if tok[:-1].lower() in self._params.abbrev_types or tok[:-1].lower().split('-')[-1] in self._params.abbrev_types:
            aug_tok.abbr = True
        else:
            aug_tok.sentbreak = True
    return