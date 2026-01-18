import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def _annotate_second_pass(self, tokens: Iterator[PunktToken]) -> Iterator[PunktToken]:
    """
        Performs a token-based classification (section 4) over the given
        tokens, making use of the orthographic heuristic (4.1.1), collocation
        heuristic (4.1.2) and frequent sentence starter heuristic (4.1.3).
        """
    for token1, token2 in _pair_iter(tokens):
        self._second_pass_annotation(token1, token2)
        yield token1