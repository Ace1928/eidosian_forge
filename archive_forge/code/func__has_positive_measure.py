import re
from nltk.stem.api import StemmerI
def _has_positive_measure(self, stem):
    return self._measure(stem) > 0