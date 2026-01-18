import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
def _word_tokenizer_re(self):
    """Compiles and returns a regular expression for word tokenization"""
    try:
        return self._re_word_tokenizer
    except AttributeError:
        self._re_word_tokenizer = re.compile(self._word_tokenize_fmt % {'NonWord': self._re_non_word_chars, 'MultiChar': self._re_multi_char_punct, 'WordStart': self._re_word_start}, re.UNICODE | re.VERBOSE)
        return self._re_word_tokenizer