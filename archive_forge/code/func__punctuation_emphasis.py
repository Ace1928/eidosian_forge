import math
import re
import string
from itertools import product
import nltk.data
from nltk.util import pairwise
def _punctuation_emphasis(self, sum_s, text):
    ep_amplifier = self._amplify_ep(text)
    qm_amplifier = self._amplify_qm(text)
    punct_emph_amplifier = ep_amplifier + qm_amplifier
    return punct_emph_amplifier