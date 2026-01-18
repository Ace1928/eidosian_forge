import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def _is_potential_sent_starter(self, cur_tok, prev_tok):
    """
        Returns True given a token and the token that precedes it if it
        seems clear that the token is beginning a sentence.
        """
    return prev_tok.sentbreak and (not (prev_tok.is_number or prev_tok.is_initial)) and cur_tok.is_alpha