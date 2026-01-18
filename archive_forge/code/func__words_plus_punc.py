import math
import re
import string
from itertools import product
import nltk.data
from nltk.util import pairwise
def _words_plus_punc(self):
    """
        Returns mapping of form:
        {
            'cat,': 'cat',
            ',cat': 'cat',
        }
        """
    no_punc_text = self.REGEX_REMOVE_PUNCTUATION.sub('', self.text)
    words_only = no_punc_text.split()
    words_only = {w for w in words_only if len(w) > 1}
    punc_before = {''.join(p): p[1] for p in product(self.PUNC_LIST, words_only)}
    punc_after = {''.join(p): p[0] for p in product(words_only, self.PUNC_LIST)}
    words_punc_dict = punc_before
    words_punc_dict.update(punc_after)
    return words_punc_dict