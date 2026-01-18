import math
import re
import string
from itertools import product
import nltk.data
from nltk.util import pairwise
def _amplify_qm(self, text):
    qm_count = text.count('?')
    qm_amplifier = 0
    if qm_count > 1:
        if qm_count <= 3:
            qm_amplifier = qm_count * 0.18
        else:
            qm_amplifier = 0.96
    return qm_amplifier