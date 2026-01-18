from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import re
import sys
import warnings
def _exact_word_regex(word):
    allowed_chars = ['\\(', '\\)', '\\[', '\\]', '\\.', ',', '!', '\\?', ' ', '^', '$']
    wildcard = '(' + '|'.join(allowed_chars) + ')+'
    return '{wildcard}{word}{wildcard}'.format(wildcard=wildcard, word=re.escape(word))