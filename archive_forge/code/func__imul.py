from collections import defaultdict
from itertools import chain
def _imul(d1, d2):
    if hasattr(d2, 'keys'):
        for k in set(chain(d1.keys(), d2.keys())):
            d1[k] = d1[k] * d2[k]
    else:
        for k in d1:
            d1[k] *= d2