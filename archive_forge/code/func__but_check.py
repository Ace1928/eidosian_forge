import math
import re
import string
from itertools import product
import nltk.data
from nltk.util import pairwise
def _but_check(self, words_and_emoticons, sentiments):
    words_and_emoticons = [w_e.lower() for w_e in words_and_emoticons]
    but = {'but'} & set(words_and_emoticons)
    if but:
        bi = words_and_emoticons.index(next(iter(but)))
        for sidx, sentiment in enumerate(sentiments):
            if sidx < bi:
                sentiments[sidx] = sentiment * 0.5
            elif sidx > bi:
                sentiments[sidx] = sentiment * 1.5
    return sentiments