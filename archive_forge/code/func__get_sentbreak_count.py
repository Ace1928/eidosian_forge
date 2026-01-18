import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def _get_sentbreak_count(self, tokens):
    """
        Returns the number of sentence breaks marked in a given set of
        augmented tokens.
        """
    return sum((1 for aug_tok in tokens if aug_tok.sentbreak))