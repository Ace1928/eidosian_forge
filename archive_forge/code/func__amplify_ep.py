import math
import re
import string
from itertools import product
import nltk.data
from nltk.util import pairwise
def _amplify_ep(self, text):
    ep_count = text.count('!')
    if ep_count > 4:
        ep_count = 4
    ep_amplifier = ep_count * 0.292
    return ep_amplifier